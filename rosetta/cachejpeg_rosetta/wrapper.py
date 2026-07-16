from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.cachejpeg.wrapper import _ensure_homo_imports
from rosetta.model.projector import load_projector
from rosetta.utils.evaluate import apply_generation_config, load_hf_model, set_default_chat_template

from .config import CacheJPEGRosettaEvalConfig, resolve_cachejpeg_rosetta_eval_config
from .fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge


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
        self.codec = codec_cls()
        self.codec_config = {
            **codec_config.get("codec", {}),
            "homo_c2c_kv_src": self.eval_codec_config.homo_c2c_kv_src,
        }
        self.fuser_bridge = RosettaFuserBridge(assets)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

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

    def prefill_on_sharer(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def prefill_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def encode_cache(self, past_key_values):
        legacy_cache = self._cache_to_codec_dtype(self._to_legacy_cache(past_key_values))
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

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        sharer_outputs = self.prefill_on_sharer(input_ids=input_ids, attention_mask=attention_mask)
        receiver_seed_outputs = self.prefill_on_receiver(input_ids=input_ids, attention_mask=attention_mask)
        payload = self.encode_cache(sharer_outputs.past_key_values)
        decoded_teacher_cache = self.decode_cache(payload)
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([input_ids, generated], dim=1)


def load_cachejpeg_rosetta_model(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    cfg = dict(model_config.get("cachejpeg_rosetta_config") or {})
    assets = _load_rosetta_assets(model_config, eval_config, device=device, generation_config=generation_config)
    wrapper = CacheJPEGRosettaEvalWrapper(
        assets=assets,
        codec_config=cfg,
    )
    return wrapper, assets.base_tokenizer
