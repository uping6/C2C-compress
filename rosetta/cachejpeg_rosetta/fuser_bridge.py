from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class LoadedRosettaAssets:
    base_model: Any
    base_tokenizer: Any
    teacher_model: Any
    teacher_tokenizer: Any
    projector_list: list[Any]
    projector_dict: dict[Any, Any]
    base_model_idx: int = 0
    teacher_model_idx: int = 1
    multi_source_fusion_mode: str = "parallel"


class RosettaFuserBridge:
    """
    Bridge object for the future teacher-cache -> base-cache fusion path.

    This skeleton keeps the receiver/fuser boundary explicit without altering the
    existing RosettaModel implementation.
    """

    def __init__(self, assets: LoadedRosettaAssets):
        self.assets = assets
        self.projector_dict = self._convert_dict_keys_to_ints(assets.projector_dict)

    def fuse_teacher_cache_to_base(self, teacher_cache, base_seed_cache=None):
        """
        Convert decoded teacher cache into base-model cache via Rosetta projectors.
        """
        teacher_cache = self._to_dynamic_cache(teacher_cache)
        base_cache = self._build_base_cache_template(teacher_cache, base_seed_cache)

        source_model_idx = int(self.assets.teacher_model_idx)
        target_model_idx = int(self.assets.base_model_idx)
        if target_model_idx not in self.projector_dict:
            return base_cache
        if source_model_idx not in self.projector_dict[target_model_idx]:
            return base_cache

        layer_map = self.projector_dict[target_model_idx][source_model_idx]
        fused_cache = self._clone_dynamic_cache(base_cache)

        with torch.no_grad():
            for target_layer_idx, entry in layer_map.items():
                pair_list = entry if isinstance(entry, list) else [entry]
                if len(pair_list) == 0:
                    continue

                base_key_cache = fused_cache.key_cache[int(target_layer_idx)]
                base_value_cache = fused_cache.value_cache[int(target_layer_idx)]
                base_kv = (
                    base_key_cache,
                    base_value_cache,
                )

                # Preserve the existing Rosetta behavior: when multiple projector
                # candidates map to one target layer, use the first result.
                source_layer_idx, projector_idx = self._normalize_pair(pair_list[0])
                source_key_cache = teacher_cache.key_cache[source_layer_idx].to(
                    device=base_key_cache.device,
                    dtype=base_key_cache.dtype,
                )
                source_value_cache = teacher_cache.value_cache[source_layer_idx].to(
                    device=base_value_cache.device,
                    dtype=base_value_cache.dtype,
                )
                source_kv = (
                    source_key_cache,
                    source_value_cache,
                )

                projector = self._prepare_projector(self.assets.projector_list[projector_idx], source_key_cache)
                projected_key, projected_value = projector.forward(source_kv, base_kv)

                fused_cache.key_cache[int(target_layer_idx)] = projected_key.to(
                    device=base_key_cache.device,
                    dtype=base_key_cache.dtype,
                ).contiguous()
                fused_cache.value_cache[int(target_layer_idx)] = projected_value.to(
                    device=base_value_cache.device,
                    dtype=base_value_cache.dtype,
                ).contiguous()

        return fused_cache

    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        if isinstance(obj, dict):
            converted = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip("-").isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                converted[new_key] = RosettaFuserBridge._convert_dict_keys_to_ints(value)
            return converted
        if isinstance(obj, list):
            return [RosettaFuserBridge._convert_dict_keys_to_ints(item) for item in obj]
        return obj

    @staticmethod
    def _to_dynamic_cache(cache: Any) -> DynamicCache:
        if isinstance(cache, DynamicCache):
            return cache
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        return DynamicCache.from_legacy_cache(tuple((key, value) for key, value in cache))

    @staticmethod
    def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
        cloned = DynamicCache()
        for key, value in zip(cache.key_cache, cache.value_cache):
            cloned.key_cache.append(key.clone().detach())
            cloned.value_cache.append(value.clone().detach())
        return cloned

    @staticmethod
    def _normalize_pair(pair: Any) -> tuple[int, int]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid projector mapping entry: {pair!r}")
        return int(pair[0]), int(pair[1])

    @staticmethod
    def _prepare_projector(projector: Any, source_key_cache: torch.Tensor):
        projector = projector.to(device=source_key_cache.device, dtype=source_key_cache.dtype)
        projector.eval()
        return projector

    def _build_base_cache_template(self, teacher_cache: DynamicCache, base_seed_cache: Any | None) -> DynamicCache:
        if base_seed_cache is not None:
            return self._clone_dynamic_cache(self._to_dynamic_cache(base_seed_cache))

        base_model = self.assets.base_model
        config = getattr(base_model, "config", None)
        if config is None:
            raise ValueError("Base model config is required to synthesize a receiver cache template.")

        num_layers = int(getattr(config, "num_hidden_layers"))
        num_kv_heads = int(
            getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads"))
        )
        head_dim = int(
            getattr(
                config,
                "head_dim",
                getattr(config, "hidden_size") // int(getattr(config, "num_attention_heads")),
            )
        )

        if len(teacher_cache.key_cache) == 0:
            raise ValueError("Teacher cache is empty; cannot synthesize receiver cache template.")

        ref_key = teacher_cache.key_cache[0]
        batch_size = int(ref_key.shape[0])
        seq_len = int(ref_key.shape[2])
        device = ref_key.device
        dtype = ref_key.dtype

        synthesized = DynamicCache()
        for _ in range(num_layers):
            synthesized.key_cache.append(
                torch.zeros((batch_size, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
            )
            synthesized.value_cache.append(
                torch.zeros((batch_size, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)
            )
        return synthesized
