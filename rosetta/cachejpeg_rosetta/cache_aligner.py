from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from .fuser_bridge import LoadedRosettaAssets
from .pre_rope import apply_receiver_compact_rope


@dataclass(frozen=True)
class LCFLatentRouting:
    routes: tuple[tuple[int, int, int], ...]
    latent_dim: int
    sequence_length: int


class ConcatCacheAligner:
    """LCF-first Sharer downsample and Receiver upsample for a causal prefix."""

    def __init__(self, assets: LoadedRosettaAssets):
        self.assets = assets
        self.projector_dict = self._convert_dict_keys_to_ints(assets.projector_dict)
        self.last_alignment_stats: dict[str, Any] | None = None

    def prepare_routing(self) -> LCFLatentRouting:
        """Validate the one-to-one layer map before streamed tensors arrive."""

        source_model_index = int(self.assets.teacher_model_idx)
        target_model_index = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_model_index][source_model_index]
        except KeyError as error:
            raise ValueError("No sharer-to-receiver projector mapping is configured.") from error
        routes = []
        latent_dim = None
        for target_layer_index, entry in layer_map.items():
            pairs = entry if isinstance(entry, list) else [entry]
            if len(pairs) != 1:
                raise ValueError(
                    "concat cache alignment requires exactly one sharer layer per receiver layer."
                )
            source_layer_index, projector_index = self._normalize_pair(pairs[0])
            projector = self.assets.projector_list[projector_index]
            if projector.__class__.__name__ != "LCFFirstProjector":
                raise TypeError(
                    "concat LCF-first alignment requires LCFFirstProjector checkpoints, "
                    f"got {projector.__class__.__name__}."
                )
            latent_dim = int(projector.latent_dim)
            routes.append(
                (int(target_layer_index), source_layer_index, projector_index)
            )
        expected = set(range(int(self.assets.base_model.config.num_hidden_layers)))
        actual = {route[0] for route in routes}
        if actual != expected:
            raise ValueError(
                f"Concat alignment is missing receiver layers: {sorted(expected - actual)}."
            )
        if len({route[1] for route in routes}) != len(routes):
            raise ValueError("Streamed concat requires one route per Sharer source layer.")
        return LCFLatentRouting(tuple(sorted(routes)), int(latent_dim), 0)

    def prepare_projectors(self, device: torch.device, dtype: torch.dtype) -> None:
        """Move read-only projectors once, before concurrent layer workers start."""

        routing = self.prepare_routing()
        for _target, _source, projector_index in routing.routes:
            self.assets.projector_list[projector_index].to(
                device=device, dtype=dtype
            ).eval()

    def encode_layer(
        self,
        route: tuple[int, int, int],
        source_key: torch.Tensor,
        source_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_layer, _source_layer, projector_index = route
        del target_layer
        projector = self.assets.projector_list[projector_index]
        with torch.no_grad():
            latent = projector.encode((source_key, source_value))
        if latent.shape[-1] % 2:
            raise ValueError("LCF-first transport requires an even latent dimension.")
        key_latent, value_latent = latent.chunk(2, dim=-1)
        return key_latent.unsqueeze(1).contiguous(), value_latent.unsqueeze(1).contiguous()

    def decode_layer(
        self,
        route: tuple[int, int, int],
        key_latent: torch.Tensor,
        value_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _target_layer, _source_layer, projector_index = route
        if key_latent.shape[1] != 1 or key_latent.shape != value_latent.shape:
            raise ValueError("LCF-first pseudo K/V must have one head and matching shapes.")
        latent = torch.cat(
            [key_latent.squeeze(1), value_latent.squeeze(1)], dim=-1
        )
        with torch.no_grad():
            key, value = self.assets.projector_list[projector_index].decode(latent)
        receiver_parameter = next(self.assets.base_model.parameters())
        return (
            key.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
            value.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
        )

    def assemble_receiver_cache(
        self,
        decoded_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]],
        routing: LCFLatentRouting,
    ) -> DynamicCache:
        prefix = DynamicCache()
        for layer_index in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = decoded_by_layer[layer_index]
            prefix.key_cache.append(key.contiguous())
            prefix.value_cache.append(value.contiguous())
        prefix = apply_receiver_compact_rope(self.assets.base_model, prefix)
        self.last_alignment_stats = {
            "alignment_type": "concat",
            "codec_order": "lcf_down_cachejpeg_lcf_up",
            "rope_mode": "pre_rope",
            "prefix_tokens": int(prefix.key_cache[0].shape[2]),
            "latent_dim": routing.latent_dim,
            "routes": [list(route) for route in routing.routes],
        }
        return prefix

    def encode(self, sharer_cache: Any) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], LCFLatentRouting]:
        """Downsample pre-RoPE Sharer K/V and expose latent halves as pseudo K/V."""

        sharer_cache = self._to_dynamic_cache(sharer_cache)
        source_model_index = int(self.assets.teacher_model_idx)
        target_model_index = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_model_index][source_model_index]
        except KeyError as error:
            raise ValueError("No sharer-to-receiver projector mapping is configured.") from error

        num_receiver_layers = int(self.assets.base_model.config.num_hidden_layers)
        latent_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        routes: list[tuple[int, int, int]] = []
        latent_dim: int | None = None
        sequence_length: int | None = None
        with torch.no_grad():
            for target_layer_index, entry in layer_map.items():
                pairs = entry if isinstance(entry, list) else [entry]
                if len(pairs) != 1:
                    raise ValueError(
                        "concat cache alignment requires exactly one sharer layer per receiver layer."
                    )
                source_layer_index, projector_index = self._normalize_pair(pairs[0])
                source_key = sharer_cache.key_cache[source_layer_index]
                source_value = sharer_cache.value_cache[source_layer_index]
                projector = self.assets.projector_list[projector_index].to(
                    device=source_key.device, dtype=source_key.dtype
                ).eval()
                if projector.__class__.__name__ != "LCFFirstProjector":
                    raise TypeError(
                        "concat LCF-first alignment requires LCFFirstProjector checkpoints, "
                        f"got {projector.__class__.__name__}."
                    )
                latent = projector.encode((source_key, source_value))
                if latent.shape[-1] % 2:
                    raise ValueError("LCF-first transport requires an even latent dimension.")
                key_latent, value_latent = latent.chunk(2, dim=-1)
                latent_by_layer[int(target_layer_index)] = (
                    key_latent.unsqueeze(1).contiguous(),
                    value_latent.unsqueeze(1).contiguous(),
                )
                latent_dim = int(latent.shape[-1])
                sequence_length = int(latent.shape[1])
                routes.append((int(target_layer_index), int(source_layer_index), int(projector_index)))

        expected_layers = set(range(num_receiver_layers))
        if set(latent_by_layer) != expected_layers:
            missing = sorted(expected_layers.difference(latent_by_layer))
            raise ValueError(f"Concat alignment is missing receiver layers: {missing}.")
        assert latent_dim is not None and sequence_length is not None
        pseudo_cache = tuple(latent_by_layer[layer] for layer in range(num_receiver_layers))
        return pseudo_cache, LCFLatentRouting(
            routes=tuple(sorted(routes)),
            latent_dim=latent_dim,
            sequence_length=sequence_length,
        )

    def decode(
        self,
        decoded_latent_cache: Any,
        routing: LCFLatentRouting,
    ) -> DynamicCache:
        """Upsample reconstructed latents, then apply Receiver RoPE to prefix K."""

        latent_cache = self._to_dynamic_cache(decoded_latent_cache)
        decoded_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        with torch.no_grad():
            for target_layer, _source_layer, projector_index in routing.routes:
                key_latent = latent_cache.key_cache[target_layer]
                value_latent = latent_cache.value_cache[target_layer]
                if key_latent.shape[1] != 1 or key_latent.shape != value_latent.shape:
                    raise ValueError("LCF-first pseudo K/V must have one head and matching shapes.")
                latent = torch.cat(
                    [key_latent.squeeze(1), value_latent.squeeze(1)], dim=-1
                )
                projector = self.assets.projector_list[projector_index].to(
                    device=latent.device, dtype=latent.dtype
                ).eval()
                key, value = projector.decode(latent)
                receiver_dtype = next(self.assets.base_model.parameters()).dtype
                receiver_device = next(self.assets.base_model.parameters()).device
                decoded_by_layer[target_layer] = (
                    key.to(device=receiver_device, dtype=receiver_dtype).contiguous(),
                    value.to(device=receiver_device, dtype=receiver_dtype).contiguous(),
                )

        prefix = DynamicCache()
        for layer_index in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = decoded_by_layer[layer_index]
            prefix.key_cache.append(key)
            prefix.value_cache.append(value)
        prefix = apply_receiver_compact_rope(self.assets.base_model, prefix)
        self.last_alignment_stats = {
            "alignment_type": "concat",
            "codec_order": "lcf_down_cachejpeg_lcf_up",
            "rope_mode": "pre_rope",
            "prefix_tokens": int(prefix.key_cache[0].shape[2]),
            "latent_dim": routing.latent_dim,
            "routes": [list(route) for route in routing.routes],
        }
        return prefix

    def align(self, sharer_cache: Any) -> DynamicCache:
        """Uncompressed convenience path used by diagnostics and unit tests."""

        latent_cache, routing = self.encode(sharer_cache)
        return self.decode(latent_cache, routing)

    @staticmethod
    def _normalize_pair(pair: Any) -> tuple[int, int]:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"Invalid projector mapping entry: {pair!r}")
        return int(pair[0]), int(pair[1])

    @staticmethod
    def _to_dynamic_cache(cache: Any) -> DynamicCache:
        if isinstance(cache, DynamicCache):
            return cache
        if hasattr(cache, "to_legacy_cache"):
            cache = cache.to_legacy_cache()
        return DynamicCache.from_legacy_cache(tuple(cache))

    @staticmethod
    def _convert_dict_keys_to_ints(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                (int(key) if isinstance(key, str) and key.lstrip("-").isdigit() else key):
                ConcatCacheAligner._convert_dict_keys_to_ints(value)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [ConcatCacheAligner._convert_dict_keys_to_ints(value) for value in obj]
        return obj
