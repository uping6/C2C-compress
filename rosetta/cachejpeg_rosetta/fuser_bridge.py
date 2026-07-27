from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from rosetta.model.latent_kv import LatentKVLayerPayload, LatentKVPayload


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
    checkpoint_dir: str | None = None


class RosettaFuserBridge:
    """
    Bridge object for the future teacher-cache -> base-cache fusion path.

    This skeleton keeps the receiver/fuser boundary explicit without altering the
    existing RosettaModel implementation.
    """

    def __init__(self, assets: LoadedRosettaAssets, adaptive_quant_table=None):
        self.assets = assets
        self.adaptive_quant_table = adaptive_quant_table
        self.projector_dict = self._convert_dict_keys_to_ints(assets.projector_dict)
        self.last_fusion_stats: dict[str, Any] | None = None

    def fuse_teacher_cache_to_base(self, teacher_cache, base_seed_cache=None):
        """
        Convert decoded teacher cache into base-model cache via Rosetta projectors.
        """
        self.last_fusion_stats = None
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
        layer_stats: list[dict[str, Any]] = []
        projected_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        quantized_source_by_target = None
        if self.adaptive_quant_table is not None:
            expected_layers = list(range(self.adaptive_quant_table.num_layers))
            actual_layers = sorted(int(layer_idx) for layer_idx in layer_map)
            if actual_layers != expected_layers:
                raise ValueError(
                    "adaptive_quant_table evaluation requires one routed source cache "
                    f"for every receiver layer; expected {expected_layers}, got {actual_layers}."
                )
            routed_source = []
            for target_layer_idx in expected_layers:
                pair_list = (
                    layer_map[target_layer_idx]
                    if isinstance(layer_map[target_layer_idx], list)
                    else [layer_map[target_layer_idx]]
                )
                if len(pair_list) != 1:
                    raise ValueError(
                        "Pre-projector adaptive quantization requires exactly one "
                        "source layer per receiver layer."
                    )
                source_layer_idx, _ = self._normalize_pair(pair_list[0])
                routed_source.append(
                    (
                        teacher_cache.key_cache[source_layer_idx],
                        teacher_cache.value_cache[source_layer_idx],
                    )
                )
            with torch.no_grad():
                quantized_source = self.adaptive_quant_table(tuple(routed_source))
            quantized_source_by_target = {
                layer_idx: reconstructed
                for layer_idx, reconstructed in enumerate(
                    quantized_source.past_key_values
                )
            }

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
                if quantized_source_by_target is not None:
                    source_key_cache, source_value_cache = quantized_source_by_target[
                        int(target_layer_idx)
                    ]
                    source_key_cache = source_key_cache.to(
                        device=base_key_cache.device, dtype=base_key_cache.dtype
                    )
                    source_value_cache = source_value_cache.to(
                        device=base_value_cache.device, dtype=base_value_cache.dtype
                    )
                else:
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

                projected_by_layer[int(target_layer_idx)] = (
                    projected_key.to(
                    device=base_key_cache.device,
                    dtype=base_key_cache.dtype,
                    ).contiguous(),
                    projected_value.to(
                    device=base_value_cache.device,
                    dtype=base_value_cache.dtype,
                    ).contiguous(),
                )

                projector_stats = getattr(projector, "last_stats", None)
                if projector_stats is not None:
                    layer_stats.append(
                        {
                            "receiver_layer": int(target_layer_idx),
                            "sharer_layer": int(source_layer_idx),
                            **dict(projector_stats),
                        }
                    )

        for target_layer_idx, (projected_key, projected_value) in projected_by_layer.items():
            fused_cache.key_cache[target_layer_idx] = projected_key
            fused_cache.value_cache[target_layer_idx] = projected_value

        if layer_stats:
            first_projector = self.assets.projector_list[0]
            self.last_fusion_stats = {
                "fusion_type": "latent_kv_joint",
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for projector in self.assets.projector_list
                    for parameter in projector.parameters()
                    if parameter.requires_grad
                ),
                "latent_dim": getattr(first_projector, "latent_dim", None),
                "joint_input_dim": getattr(first_projector, "joint_input_dim", None),
                "layers": layer_stats,
            }
            if self.adaptive_quant_table is not None:
                result = self.adaptive_quant_table.last_result
                self.last_fusion_stats["adaptive_quant_table"] = {
                    "estimated_payload_bits": float(
                        result.estimated_payload_bits.detach().item()
                    ),
                    "mean_alpha": float(result.alpha.float().mean().item()),
                }
        else:
            self.last_fusion_stats = None

        return fused_cache

    def encode_teacher_cache_to_latents(
        self,
        teacher_cache,
        *,
        move_to_cpu: bool = True,
    ) -> LatentKVPayload:
        """Sharer-side encoding that never receives or reads receiver cache state."""

        teacher_cache = self._to_dynamic_cache(teacher_cache)
        source_model_idx = int(self.assets.teacher_model_idx)
        target_model_idx = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_model_idx][source_model_idx]
        except KeyError as exc:
            raise ValueError("No sharer-to-receiver projector mapping is configured.") from exc

        payload_layers: list[LatentKVLayerPayload] = []
        latent_dim: int | None = None
        sequence_length: int | None = None
        source_dtype: str | None = None
        with torch.no_grad():
            for target_layer_idx, entry in layer_map.items():
                pair_list = entry if isinstance(entry, list) else [entry]
                if not pair_list:
                    continue
                source_layer_idx, projector_idx = self._normalize_pair(pair_list[0])
                projector = self.assets.projector_list[projector_idx]
                encode = getattr(projector, "encode", None)
                if not callable(encode):
                    raise TypeError(
                        "Split latent communication requires SplitLatentKVProjector, "
                        f"got {projector.__class__.__name__}."
                    )
                source_key = teacher_cache.key_cache[source_layer_idx]
                source_value = teacher_cache.value_cache[source_layer_idx]
                projector = self._prepare_projector(projector, source_key)
                latent = encode((source_key, source_value))
                latent_dim = int(latent.shape[-1])
                sequence_length = int(latent.shape[1])
                source_dtype = str(latent.dtype)
                wire_latent = latent.detach()
                if move_to_cpu:
                    # The uncompressed wire path serializes CPU tensors and
                    # avoids embedding CUDA device state in pickle.
                    wire_latent = wire_latent.to("cpu")
                payload_layers.append(
                    LatentKVLayerPayload(
                        receiver_layer=int(target_layer_idx),
                        sharer_layer=int(source_layer_idx),
                        projector_idx=int(projector_idx),
                        latent=wire_latent.contiguous(),
                    )
                )

        if not payload_layers or latent_dim is None or sequence_length is None:
            raise ValueError("Split latent encoding produced an empty payload.")
        return LatentKVPayload(
            layers=payload_layers,
            latent_dim=latent_dim,
            sequence_length=sequence_length,
            source_dtype=source_dtype or "unknown",
            quantized=False,
        )

    def fuse_latents_to_base(self, payload: LatentKVPayload, base_seed_cache) -> DynamicCache:
        """Receiver-side conditional decoding from transmitted sharer latents."""

        if not isinstance(payload, LatentKVPayload):
            raise TypeError(
                f"Expected LatentKVPayload, got {type(payload)!r}."
            )
        if payload.quantized:
            raise ValueError("Quantized latent payloads are not implemented yet.")
        base_cache = self._to_dynamic_cache(base_seed_cache)
        fused_cache = self._clone_dynamic_cache(base_cache)
        layer_stats: list[dict[str, Any]] = []

        with torch.no_grad():
            for layer_payload in payload.layers:
                target_layer_idx = int(layer_payload.receiver_layer)
                projector_idx = int(layer_payload.projector_idx)
                if target_layer_idx < 0 or target_layer_idx >= len(fused_cache.key_cache):
                    raise ValueError(
                        f"Receiver layer {target_layer_idx} is outside the cache range."
                    )
                if projector_idx < 0 or projector_idx >= len(self.assets.projector_list):
                    raise ValueError(
                        f"Projector index {projector_idx} is outside the loaded range."
                    )
                base_key = fused_cache.key_cache[target_layer_idx]
                base_value = fused_cache.value_cache[target_layer_idx]
                if int(base_key.shape[2]) != payload.sequence_length:
                    raise ValueError(
                        "Latent and receiver sequence lengths differ: "
                        f"latent={payload.sequence_length}, receiver={base_key.shape[2]}."
                    )
                projector = self._prepare_projector(
                    self.assets.projector_list[projector_idx], base_key
                )
                decode = getattr(projector, "decode", None)
                if not callable(decode):
                    raise TypeError(
                        "Split latent communication requires SplitLatentKVProjector, "
                        f"got {projector.__class__.__name__}."
                    )
                latent = layer_payload.latent.to(
                    device=base_key.device, dtype=base_key.dtype
                )
                fused_key, fused_value = decode(
                    latent, (base_key, base_value)
                )
                fused_cache.key_cache[target_layer_idx] = fused_key.contiguous()
                fused_cache.value_cache[target_layer_idx] = fused_value.contiguous()
                layer_stats.append(
                    {
                        "receiver_layer": target_layer_idx,
                        "sharer_layer": int(layer_payload.sharer_layer),
                        **dict(getattr(projector, "last_stats", None) or {}),
                    }
                )

        first_projector = self.assets.projector_list[payload.layers[0].projector_idx]
        self.last_fusion_stats = {
            "fusion_type": "latent_kv_split",
            "latent_dim": payload.latent_dim,
            "quantized": payload.quantized,
            "trainable_parameter_count": sum(
                parameter.numel()
                for projector in self.assets.projector_list
                for parameter in projector.parameters()
                if parameter.requires_grad
            ),
            "sharer_input_dim": (
                2
                * int(getattr(first_projector, "sharer_num_kv_heads"))
                * int(getattr(first_projector, "sharer_head_dim"))
            ),
            "layers": layer_stats,
        }
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
