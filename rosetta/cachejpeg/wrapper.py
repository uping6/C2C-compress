from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer

from rosetta.cachejpeg.config import CacheJPEGEvalConfig, resolve_cachejpeg_eval_config
from rosetta.cachejpeg.transport import build_transport
from rosetta.utils.evaluate import apply_generation_config, load_hf_model, set_default_chat_template


def _ensure_homo_imports(src_root: str):
    src_path = Path(src_root)
    if not src_path.exists():
        raise FileNotFoundError(
            f"CacheJPEG requires HomoC2C-KV sources at {src_path}, but the path does not exist."
        )
    src_text = str(src_path)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

    from homo_c2c_kv.cache.interop import to_dynamic_cache, to_legacy_cache
    from homo_c2c_kv.codec.cachejpeg.codec import CacheJPEGCodec
    from homo_c2c_kv.codec.cachejpeg.config import resolve_cachejpeg_config
    from rosetta.cachejpeg.entropy_backends import install_homo_cachejpeg_entropy_backends

    install_homo_cachejpeg_entropy_backends()

    return CacheJPEGCodec, resolve_cachejpeg_config, to_dynamic_cache, to_legacy_cache


class CacheJPEGEvalWrapper:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        codec_config: dict[str, Any],
        codec: Any | None = None,
        transport: Any | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_codec_config = resolve_cachejpeg_eval_config(codec_config)
        (
            codec_cls,
            _codec_config_resolver,
            self._to_dynamic_cache,
            self._to_legacy_cache,
        ) = _ensure_homo_imports(self.eval_codec_config.homo_c2c_kv_src)
        if codec is not None:
            self.codec = codec
        elif self.eval_codec_config.compute.backend == "gpu":
            from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

            self.codec = GPUCacheJPEGCodec(device=next(model.parameters()).device)
        else:
            self.codec = codec_cls()
        # CacheJPEGCodec.encode/decode resolve the raw mapping internally. Passing
        # the already-resolved CacheJPEGConfig here makes the codec try to call
        # ``.get`` on a dataclass and fails for every evaluated sample.
        self.codec_config = dict(codec_config)
        self.transport = transport if transport is not None else build_transport(
            dict(codec_config.get("transport") or {})
        )
        self.last_transport_stats = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _build_position_ids(self, attention_mask: Optional[torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
        if attention_mask is None:
            return torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
        position_ids = attention_mask.long().cumsum(-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    def _decode_from_cache(
        self,
        last_token: torch.Tensor,
        past_key_values,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        generated = []
        current_input = last_token
        current_past = self._to_dynamic_cache(self._cache_to_model_dtype(past_key_values))
        for _ in range(max(1, int(max_new_tokens))):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=current_past,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            if do_sample:
                scaled = logits if temperature <= 0 else logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            current_input = next_token
            current_past = self._to_dynamic_cache(outputs.past_key_values)
            if self.tokenizer.eos_token_id is not None and next_token.item() == self.tokenizer.eos_token_id:
                break
        if not generated:
            return last_token
        return torch.cat(generated, dim=1)

    def _cache_to_model_dtype(self, past_key_values):
        parameter = next(self.model.parameters())
        legacy_cache = self._to_legacy_cache(past_key_values)
        return tuple(
            (
                key.to(device=parameter.device, dtype=parameter.dtype),
                value.to(device=parameter.device, dtype=parameter.dtype),
            )
            for key, value in legacy_cache
        )

    @staticmethod
    def _cache_to_codec_dtype(past_key_values):
        return tuple(
            (
                key.detach().to(dtype=torch.float32),
                value.detach().to(dtype=torch.float32),
            )
            for key, value in past_key_values
        )

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        position_ids = self._build_position_ids(attention_mask, input_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        legacy_cache = self._to_legacy_cache(outputs.past_key_values)
        if not getattr(self.codec, "uses_gpu_transform", False):
            legacy_cache = self._cache_to_codec_dtype(legacy_cache)
        payload = self.codec.encode(legacy_cache, self.codec_config)
        received_payload = self.transport.roundtrip(payload) if self.transport is not None else payload
        self.last_transport_stats = self.transport.last_stats if self.transport is not None else None
        reconstructed = self.codec.decode(received_payload, self.codec_config)
        generated = self._decode_from_cache(
            last_token=input_ids[:, -1:],
            past_key_values=reconstructed,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([input_ids, generated], dim=1)


def load_cachejpeg_model(
    model_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    rosetta_config = model_config.get("rosetta_config") or {}
    base_model_name = (
        model_config.get("base_model_name")
        or model_config.get("base_model")
        or rosetta_config.get("base_model")
    )
    if not base_model_name:
        raise ValueError(
            "CacheJPEG evaluation requires model.base_model_name, model.base_model, "
            "or model.rosetta_config.base_model."
        )

    base_model, tokenizer = load_hf_model(base_model_name, device=device, generation_config=generation_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    set_default_chat_template(tokenizer, str(base_model_name))
    apply_generation_config(base_model, generation_config)

    codec_config = dict(model_config.get("cachejpeg_config") or {})
    codec_config.setdefault("method", "cachejpeg")
    codec_config.setdefault("base_model", str(base_model_name))
    if rosetta_config.get("teacher_model") is not None:
        codec_config.setdefault("teacher_model", rosetta_config.get("teacher_model"))
    if rosetta_config.get("checkpoints_dir") is not None:
        codec_config.setdefault("checkpoints_dir", rosetta_config.get("checkpoints_dir"))
    if codec_config["method"] == "cachejpeg":
        # HomoC2C-KV's current codec resolver expects the JPEG-style cache path under this method name.
        codec_config["method"] = "cachejpeg_norm_quant"
    wrapper = CacheJPEGEvalWrapper(
        model=base_model,
        tokenizer=tokenizer,
        codec_config=codec_config,
    )
    return wrapper, tokenizer
