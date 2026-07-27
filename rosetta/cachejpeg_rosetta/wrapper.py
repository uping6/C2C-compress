from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.cachejpeg.wrapper import _ensure_homo_imports
from rosetta.cachejpeg.transport import build_transport
from rosetta.cachejpeg.transport import serialize_payload
from rosetta.model.projector import load_projector
from rosetta.utils.evaluate import apply_generation_config, load_hf_model, set_default_chat_template

from .config import CacheJPEGRosettaEvalConfig, resolve_cachejpeg_rosetta_eval_config
from .fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge
from .layer_streaming import LayerCompressionPipeline, LayerPrefillTimer, StreamingDynamicCache


def _hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


def _resolve_checkpoint_dir(checkpoints_dir: str, checkpoint_subfolder: Optional[str]) -> str:
    if not checkpoint_subfolder:
        return checkpoints_dir
    candidate = os.path.join(checkpoints_dir, checkpoint_subfolder)
    return candidate if os.path.isdir(candidate) else checkpoints_dir


def _load_projector_assets(checkpoint_dir: str) -> tuple[list[Any], dict[Any, Any]]:
    projector_list = []
    if os.path.isdir(checkpoint_dir):
        num_projectors = len(
            [f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)]
        )
        for proj_idx in range(num_projectors):
            json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
            pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
            proj = load_projector(json_cfg)
            state_dict = torch.load(pt_path, map_location="cpu")
            proj.load_state_dict(state_dict, strict=False)
            projector_list.append(proj)

        projector_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        if os.path.isfile(projector_cfg_path):
            import json

            with open(projector_cfg_path, "r", encoding="utf-8") as f:
                projector_dict = json.load(f)
        else:
            projector_dict = {}
    else:
        projector_dict = {}
    return projector_list, projector_dict


def _load_rosetta_assets(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> LoadedRosettaAssets:
    rosetta_config = model_config.get("rosetta_config") or {}
    base_model_name = rosetta_config["base_model"]
    teacher_model_name = rosetta_config["teacher_model"]
    checkpoint_dir = _resolve_checkpoint_dir(
        rosetta_config["checkpoints_dir"],
        eval_config.get("rosetta_checkpoint_subfolder"),
    )

    base_model, base_tokenizer = load_hf_model(base_model_name, device=device, generation_config=generation_config)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=getattr(base_model, "dtype", None),
        local_files_only=_hf_local_files_only(),
    ).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        local_files_only=_hf_local_files_only(),
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    set_default_chat_template(teacher_tokenizer, str(teacher_model_name))
    apply_generation_config(teacher_model, generation_config)

    projector_list, projector_dict = _load_projector_assets(checkpoint_dir)
    return LoadedRosettaAssets(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        projector_list=projector_list,
        projector_dict=projector_dict,
    )


class CacheJPEGRosettaEvalWrapper:
    """
    Skeleton for:
      teacher/sharer prefill -> cachejpeg encode/decode -> fuser -> base/receiver generate.
    """

    def __init__(
        self,
        assets: LoadedRosettaAssets,
        codec_config: dict[str, Any],
    ):
        self.assets = assets
        self.base_model = assets.base_model
        self.base_tokenizer = assets.base_tokenizer
        self.teacher_model = assets.teacher_model
        self.teacher_tokenizer = assets.teacher_tokenizer
        self.eval_codec_config: CacheJPEGRosettaEvalConfig = resolve_cachejpeg_rosetta_eval_config(codec_config)
        (
            codec_cls,
            _codec_config_resolver,
            self._to_dynamic_cache,
            self._to_legacy_cache,
        ) = _ensure_homo_imports(self.eval_codec_config.homo_c2c_kv_src)
        self.codec_config = {
            **codec_config.get("codec", {}),
            "homo_c2c_kv_src": self.eval_codec_config.homo_c2c_kv_src,
        }
        if self.eval_codec_config.codec.compute.backend == "gpu":
            from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

            self.codec = GPUCacheJPEGCodec(device=next(self.teacher_model.parameters()).device)
        else:
            self.codec = codec_cls()
        self.fuser_bridge = RosettaFuserBridge(assets)
        transport_config = codec_config.get("transport") or (codec_config.get("codec") or {}).get("transport")
        self.transport = build_transport(dict(transport_config or {}))
        self.last_transport_stats = None
        self.last_codec_stats: dict[str, Any] | None = None
        self.ablation_config = dict(codec_config.get("ablation") or {})
        if self.eval_codec_config.layer_streaming.enabled and not hasattr(self.codec, "encode_layer"):
            raise ValueError(
                "cachejpeg_rosetta.layer_streaming currently requires compute.backend=gpu."
            )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @staticmethod
    def _split_aligned_inputs(input_ids, attention_mask=None):
        """Split TokenAligner output into receiver and sharer inputs."""
        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) != 2:
                raise ValueError(
                    "CacheJPEG-Rosetta aligned input_ids must contain base and teacher tensors"
                )
            base_input_ids, teacher_input_ids = input_ids
            if isinstance(attention_mask, (list, tuple)):
                if len(attention_mask) != 2:
                    raise ValueError(
                        "CacheJPEG-Rosetta aligned attention_mask must contain base and teacher tensors"
                    )
                base_attention_mask, teacher_attention_mask = attention_mask
            else:
                base_attention_mask = teacher_attention_mask = attention_mask
            return base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask
        return input_ids, input_ids, attention_mask, attention_mask

    def _decode_from_receiver_cache(
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
        current_past = self._to_dynamic_cache(past_key_values)
        for _ in range(max(1, int(max_new_tokens))):
            with torch.no_grad():
                outputs = self.base_model(
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
            if self.base_tokenizer.eos_token_id is not None and next_token.item() == self.base_tokenizer.eos_token_id:
                break
        if not generated:
            return last_token
        return torch.cat(generated, dim=1)

    def prefill_on_sharer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
    ):
        model_kwargs = {}
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
        with torch.no_grad():
            return self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **model_kwargs,
            )

    def prefill_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def encode_cache(self, past_key_values):
        legacy_cache = self._to_legacy_cache(past_key_values)
        if not getattr(self.codec, "uses_gpu_transform", False):
            legacy_cache = self._cache_to_codec_dtype(legacy_cache)
        return self.codec.encode(legacy_cache, self.codec_config)

    def decode_cache(self, payload):
        return self.codec.decode(payload, self.codec_config)

    @staticmethod
    def _cache_to_codec_dtype(past_key_values):
        return tuple(
            (
                key.detach().to(dtype=torch.float32),
                value.detach().to(dtype=torch.float32),
            )
            for key, value in past_key_values
        )

    def fuse_to_receiver_cache(self, decoded_teacher_cache, base_seed_cache=None):
        return self.fuser_bridge.fuse_teacher_cache_to_base(
            decoded_teacher_cache,
            base_seed_cache=base_seed_cache,
        )

    def generate_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    def _generate_with_layer_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_aligned_inputs(input_ids, attention_mask)
        )
        num_layers = int(self.teacher_model.config.num_hidden_layers)
        pipeline = LayerCompressionPipeline(
            codec=self.codec,
            codec_config=self.codec_config,
            transport=self.transport,
            num_layers=num_layers,
            queue_size=self.eval_codec_config.layer_streaming.queue_size,
        )
        streaming_cache = StreamingDynamicCache(pipeline.submit)
        prefill_timer = LayerPrefillTimer(self.teacher_model, num_layers)
        prefill_timer.start()
        pipeline_started = time.perf_counter()
        try:
            self.prefill_on_sharer(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                past_key_values=streaming_cache,
            )
            # Receiver prefill overlaps with any compression work still queued.
            receiver_seed_outputs = self.prefill_on_receiver(
                input_ids=base_input_ids, attention_mask=base_attention_mask
            )
            decoded_teacher_cache = pipeline.finish()
            teacher_device = next(self.teacher_model.parameters()).device
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            layer_prefill_seconds = prefill_timer.finish()
        except BaseException:
            for handle in prefill_timer.handles:
                handle.remove()
            prefill_timer.handles.clear()
            pipeline.abort()
            raise
        pipeline_seconds = time.perf_counter() - pipeline_started
        self.last_transport_stats = pipeline.aggregate_transport_stats()

        zero_sharer_cache = bool(self.ablation_config.get("zero_sharer_cache_at_receiver", False))
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in decoded_teacher_cache
            )
        self.last_codec_stats = {
            "mode": "layer_streaming",
            "num_layers": num_layers,
            "queue_size": self.eval_codec_config.layer_streaming.queue_size,
            "original_kv_bytes": pipeline.original_kv_bytes,
            "payload_bytes": pipeline.payload_bytes,
            "compression_factor": (
                float(pipeline.original_kv_bytes / pipeline.payload_bytes)
                if pipeline.payload_bytes
                else 0.0
            ),
            "space_saving_ratio": (
                float(1.0 - pipeline.payload_bytes / pipeline.original_kv_bytes)
                if pipeline.original_kv_bytes
                else 0.0
            ),
            "encode_seconds": float(pipeline.encode_seconds),
            "avg_layer_encode_seconds": float(
                sum(value for value in pipeline.layer_encode_seconds if value is not None)
                / max(1, sum(value is not None for value in pipeline.layer_encode_seconds))
            ),
            "layer_encode_seconds": [
                float(value) if value is not None else None
                for value in pipeline.layer_encode_seconds
            ],
            "avg_layer_prefill_seconds": float(sum(layer_prefill_seconds) / len(layer_prefill_seconds)),
            "layer_prefill_seconds": [float(value) for value in layer_prefill_seconds],
            "decode_seconds": float(pipeline.decode_seconds),
            "pipeline_seconds": float(pipeline_seconds),
            "transport_bandwidth_bytes_per_sec": getattr(
                self.transport, "bandwidth_bytes_per_sec", None
            ),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        streaming_config = getattr(getattr(self, "eval_codec_config", None), "layer_streaming", None)
        if streaming_config is not None and streaming_config.enabled:
            return self._generate_with_layer_streaming(
                input_ids, attention_mask, generation_config
            )
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_aligned_inputs(input_ids, attention_mask)
        )
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
        )
        receiver_seed_outputs = self.prefill_on_receiver(
            input_ids=base_input_ids, attention_mask=base_attention_mask
        )
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_started = time.perf_counter()
        payload = self.encode_cache(legacy_sharer_cache)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started
        payload_bytes = len(serialize_payload(payload))
        transport = getattr(self, "transport", None)
        received_payload = transport.roundtrip(payload) if transport is not None else payload
        self.last_transport_stats = transport.last_stats if transport is not None else None
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_started = time.perf_counter()
        decoded_teacher_cache = self.decode_cache(received_payload)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_seconds = time.perf_counter() - decode_started
        zero_sharer_cache = bool(
            getattr(self, "ablation_config", {}).get("zero_sharer_cache_at_receiver", False)
        )
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in self._to_legacy_cache(decoded_teacher_cache)
            )
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "payload_bytes": payload_bytes,
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "space_saving_ratio": float(1.0 - payload_bytes / original_kv_bytes) if original_kv_bytes else 0.0,
            "encode_seconds": float(encode_seconds),
            "decode_seconds": float(decode_seconds),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)


def load_cachejpeg_rosetta_model(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    cfg = dict(model_config.get("cachejpeg_rosetta_config") or {})
    ablation_config = dict(cfg.get("ablation") or {})
    receiver_only = bool(ablation_config.get("receiver_only", False))
    sharer_only = bool(ablation_config.get("sharer_only", False))
    if receiver_only and sharer_only:
        raise ValueError("receiver_only and sharer_only cannot both be enabled")

    rosetta_config = dict(model_config.get("rosetta_config") or {})
    if receiver_only:
        base_model_name = rosetta_config["base_model"]
        base_model, base_tokenizer = load_hf_model(
            base_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the sharer/teacher,
        # projector, fuser, CacheJPEG codec, or transport.
        return base_model, base_tokenizer
    if sharer_only:
        teacher_model_name = rosetta_config["teacher_model"]
        teacher_model, teacher_tokenizer = load_hf_model(
            teacher_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the receiver/base,
        # projector, fuser, CacheJPEG codec, or transport. Returning the teacher
        # tokenizer also ensures the LongBench prompt is tokenized in the
        # sharer's own vocabulary.
        return teacher_model, teacher_tokenizer
    assets = _load_rosetta_assets(model_config, eval_config, device=device, generation_config=generation_config)
    wrapper = CacheJPEGRosettaEvalWrapper(
        assets=assets,
        codec_config=cfg,
    )
    return wrapper, assets.base_tokenizer
