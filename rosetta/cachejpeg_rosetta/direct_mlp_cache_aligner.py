"""Uncompressed direct-MLP concat cache alignment."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from rosetta.model.direct_pre_rope_mlp import DirectPreRopeMLPProjector

from .cache_aligner import ConcatCacheAligner
from .fuser_bridge import LoadedRosettaAssets
from .pre_rope import apply_receiver_compact_rope


class DirectMLPConcatCacheAligner:
    """Project pre-RoPE Sharer KV directly into a Receiver prefix cache."""

    def __init__(self, assets: LoadedRosettaAssets):
        self.assets = assets
        self.projector_dict = ConcatCacheAligner._convert_dict_keys_to_ints(
            assets.projector_dict
        )
        self.last_alignment_stats: dict[str, Any] | None = None

    def prepare_routing(self) -> tuple[tuple[int, int, int], ...]:
        source_index = int(self.assets.teacher_model_idx)
        target_index = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_index][source_index]
        except KeyError as error:
            raise ValueError(
                "No sharer-to-receiver projector mapping is configured."
            ) from error

        routes = []
        for target_layer, entry in layer_map.items():
            pairs = entry if isinstance(entry, list) else [entry]
            if len(pairs) != 1:
                raise ValueError(
                    "Direct MLP concat requires one Sharer layer per Receiver layer."
                )
            source_layer, projector_index = ConcatCacheAligner._normalize_pair(
                pairs[0]
            )
            projector = self.assets.projector_list[projector_index]
            if not isinstance(projector, DirectPreRopeMLPProjector):
                raise TypeError(
                    "Direct MLP concat requires DirectPreRopeMLPProjector checkpoints."
                )
            routes.append((int(target_layer), source_layer, projector_index))

        expected = set(range(int(self.assets.base_model.config.num_hidden_layers)))
        actual = {route[0] for route in routes}
        if actual != expected:
            raise ValueError(
                f"Direct MLP concat is missing receiver layers: {sorted(expected - actual)}."
            )
        if len({route[1] for route in routes}) != len(routes):
            raise ValueError("Direct MLP concat requires unique Sharer source layers.")
        return tuple(sorted(routes))

    def align(self, sharer_cache: Any) -> DynamicCache:
        sharer_cache = ConcatCacheAligner._to_dynamic_cache(sharer_cache)
        routes = self.prepare_routing()
        receiver_parameter = next(self.assets.base_model.parameters())
        projected_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        sequence_length = None
        with torch.no_grad():
            for target_layer, source_layer, projector_index in routes:
                source_key = sharer_cache.key_cache[source_layer]
                source_value = sharer_cache.value_cache[source_layer]
                projector = self.assets.projector_list[projector_index].to(
                    device=source_key.device, dtype=source_key.dtype
                ).eval()
                key, value = projector.project((source_key, source_value))
                projected_by_layer[target_layer] = (
                    key.to(
                        device=receiver_parameter.device,
                        dtype=receiver_parameter.dtype,
                    ).contiguous(),
                    value.to(
                        device=receiver_parameter.device,
                        dtype=receiver_parameter.dtype,
                    ).contiguous(),
                )
                sequence_length = int(key.shape[2])

        prefix = DynamicCache()
        for layer in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = projected_by_layer[layer]
            prefix.key_cache.append(key)
            prefix.value_cache.append(value)
        prefix = apply_receiver_compact_rope(self.assets.base_model, prefix)
        self.last_alignment_stats = {
            "alignment_type": "concat",
            "concat_projector_type": "direct_pre_rope_mlp",
            "communication_mode": "local_direct",
            "codec_order": "direct_mlp",
            "rope_mode": "pre_rope",
            "prefix_tokens": int(sequence_length),
            "routes": [list(route) for route in routes],
        }
        return prefix
