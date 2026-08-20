"""Concat aligner for the LCF shared-latent learned K/V projection variant."""

from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from rosetta.model.lcf_projected_kv import LCFProjectedKVProjector

from .cache_aligner import ConcatCacheAligner, LCFLatentRouting
from .pre_rope import apply_receiver_compact_rope


class ProjectedKVConcatCacheAligner(ConcatCacheAligner):
    """LCF encode, learned 128→64 K/V transport projections, then decode."""

    def prepare_routing(self) -> LCFLatentRouting:
        source_index = int(self.assets.teacher_model_idx)
        target_index = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_index][source_index]
        except KeyError as error:
            raise ValueError("No sharer-to-receiver projector mapping is configured.") from error
        routes = []
        shared_latent_dim = None
        for target_layer, entry in layer_map.items():
            pairs = entry if isinstance(entry, list) else [entry]
            if len(pairs) != 1:
                raise ValueError(
                    "Projected-KV concat requires one Sharer layer per Receiver layer."
                )
            source_layer, projector_index = self._normalize_pair(pairs[0])
            projector = self.assets.projector_list[projector_index]
            if not isinstance(projector, LCFProjectedKVProjector):
                raise TypeError(
                    "Projected-KV concat requires LCFProjectedKVProjector checkpoints."
                )
            shared_latent_dim = projector.shared_latent_dim
            routes.append((int(target_layer), source_layer, projector_index))
        expected = set(range(int(self.assets.base_model.config.num_hidden_layers)))
        actual = {route[0] for route in routes}
        if actual != expected:
            raise ValueError(
                f"Projected-KV concat is missing receiver layers: {sorted(expected - actual)}."
            )
        if len({route[1] for route in routes}) != len(routes):
            raise ValueError("Streamed concat requires one route per Sharer source layer.")
        return LCFLatentRouting(tuple(sorted(routes)), int(shared_latent_dim), 0)

    def encode_layer(self, route, source_key, source_value):
        _target_layer, _source_layer, projector_index = route
        projector = self.assets.projector_list[projector_index]
        if not isinstance(projector, LCFProjectedKVProjector):
            raise TypeError(
                "Projected-KV concat requires LCFProjectedKVProjector checkpoints."
            )
        with torch.no_grad():
            key_latent, value_latent = projector.encode((source_key, source_value))
        return (
            key_latent.unsqueeze(1).contiguous(),
            value_latent.unsqueeze(1).contiguous(),
        )

    def decode_layer(self, route, key_latent, value_latent):
        _target_layer, _source_layer, projector_index = route
        if key_latent.shape[1] != 1 or key_latent.shape != value_latent.shape:
            raise ValueError("Projected pseudo K/V must have one head and matching shapes.")
        projector = self.assets.projector_list[projector_index]
        if not isinstance(projector, LCFProjectedKVProjector):
            raise TypeError(
                "Projected-KV concat requires LCFProjectedKVProjector checkpoints."
            )
        with torch.no_grad():
            key, value = projector.decode_transport(
                key_latent.squeeze(1), value_latent.squeeze(1)
            )
        receiver_parameter = next(self.assets.base_model.parameters())
        return (
            key.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
            value.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
        )

    def assemble_receiver_cache(self, decoded_by_layer, routing):
        prefix = DynamicCache()
        for layer in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = decoded_by_layer[layer]
            prefix.key_cache.append(key.contiguous())
            prefix.value_cache.append(value.contiguous())
        prefix = apply_receiver_compact_rope(self.assets.base_model, prefix)
        first_projector = self.assets.projector_list[routing.routes[0][2]]
        self.last_alignment_stats = {
            "alignment_type": "concat",
            "concat_projector_type": "lcf_projected_kv",
            "codec_order": "lcf_project_kv_cachejpeg_lcf_up",
            "rope_mode": "pre_rope",
            "prefix_tokens": int(prefix.key_cache[0].shape[2]),
            "shared_latent_dim": routing.latent_dim,
            "transport_dim": int(first_projector.key_latent_dim),
            "routes": [list(route) for route in routing.routes],
        }
        return prefix

    def encode(self, sharer_cache: Any):
        sharer_cache = self._to_dynamic_cache(sharer_cache)
        source_index = int(self.assets.teacher_model_idx)
        target_index = int(self.assets.base_model_idx)
        try:
            layer_map = self.projector_dict[target_index][source_index]
        except KeyError as error:
            raise ValueError("No sharer-to-receiver projector mapping is configured.") from error

        num_receiver_layers = int(self.assets.base_model.config.num_hidden_layers)
        latent_by_layer = {}
        routes = []
        shared_latent_dim = None
        sequence_length = None
        with torch.no_grad():
            for target_layer, entry in layer_map.items():
                pairs = entry if isinstance(entry, list) else [entry]
                if len(pairs) != 1:
                    raise ValueError("Projected-KV concat requires one Sharer layer per Receiver layer.")
                source_layer, projector_index = self._normalize_pair(pairs[0])
                source_key = sharer_cache.key_cache[source_layer]
                source_value = sharer_cache.value_cache[source_layer]
                projector = self.assets.projector_list[projector_index].to(
                    device=source_key.device, dtype=source_key.dtype
                ).eval()
                if not isinstance(projector, LCFProjectedKVProjector):
                    raise TypeError("Projected-KV concat requires LCFProjectedKVProjector checkpoints.")
                key_latent, value_latent = projector.encode((source_key, source_value))
                latent_by_layer[int(target_layer)] = (
                    key_latent.unsqueeze(1).contiguous(),
                    value_latent.unsqueeze(1).contiguous(),
                )
                shared_latent_dim = projector.shared_latent_dim
                sequence_length = int(key_latent.shape[1])
                routes.append((int(target_layer), source_layer, projector_index))

        expected = set(range(num_receiver_layers))
        if set(latent_by_layer) != expected:
            raise ValueError(
                f"Projected-KV concat is missing receiver layers: {sorted(expected - set(latent_by_layer))}."
            )
        pseudo_cache = tuple(latent_by_layer[layer] for layer in range(num_receiver_layers))
        return pseudo_cache, LCFLatentRouting(
            routes=tuple(sorted(routes)),
            latent_dim=int(shared_latent_dim),
            sequence_length=int(sequence_length),
        )

    def decode(self, decoded_latent_cache: Any, routing: LCFLatentRouting) -> DynamicCache:
        latent_cache = self._to_dynamic_cache(decoded_latent_cache)
        decoded_by_layer = {}
        with torch.no_grad():
            for target_layer, _source_layer, projector_index in routing.routes:
                key_latent = latent_cache.key_cache[target_layer]
                value_latent = latent_cache.value_cache[target_layer]
                if key_latent.shape[1] != 1 or key_latent.shape != value_latent.shape:
                    raise ValueError("Projected pseudo K/V must have one head and matching shapes.")
                projector = self.assets.projector_list[projector_index].to(
                    device=key_latent.device, dtype=key_latent.dtype
                ).eval()
                if not isinstance(projector, LCFProjectedKVProjector):
                    raise TypeError("Projected-KV concat requires LCFProjectedKVProjector checkpoints.")
                key, value = projector.decode_transport(
                    key_latent.squeeze(1), value_latent.squeeze(1)
                )
                receiver_parameter = next(self.assets.base_model.parameters())
                decoded_by_layer[target_layer] = (
                    key.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
                    value.to(device=receiver_parameter.device, dtype=receiver_parameter.dtype),
                )

        prefix = DynamicCache()
        for layer in range(int(self.assets.base_model.config.num_hidden_layers)):
            key, value = decoded_by_layer[layer]
            prefix.key_cache.append(key.contiguous())
            prefix.value_cache.append(value.contiguous())
        prefix = apply_receiver_compact_rope(self.assets.base_model, prefix)
        self.last_alignment_stats = {
            "alignment_type": "concat",
            "concat_projector_type": "lcf_projected_kv",
            "codec_order": "lcf_project_kv_cachejpeg_lcf_up",
            "rope_mode": "pre_rope",
            "prefix_tokens": int(prefix.key_cache[0].shape[2]),
            "shared_latent_dim": routing.latent_dim,
            "transport_dim": int(latent_cache.key_cache[0].shape[-1]),
            "routes": [list(route) for route in routing.routes],
        }
        return prefix
