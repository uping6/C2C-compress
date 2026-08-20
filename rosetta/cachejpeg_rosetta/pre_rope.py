from __future__ import annotations

import importlib
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class PreRopeLayerTask:
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    ready_event: torch.cuda.Event | None = None


class StreamingPreRopeKVPublisher:
    """Join pre-RoPE K hooks with cache V updates and publish complete layers."""

    def __init__(self, on_layer_ready):
        self.on_layer_ready = on_layer_ready
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.published: set[int] = set()

    def capture_key(self, layer_idx: int, key: torch.Tensor) -> None:
        self.keys[int(layer_idx)] = key.detach()
        self._publish_if_ready(int(layer_idx))

    def capture_value(self, layer_idx: int, value: torch.Tensor) -> None:
        self.values[int(layer_idx)] = value.detach()
        self._publish_if_ready(int(layer_idx))

    def _publish_if_ready(self, layer_idx: int) -> None:
        if (
            layer_idx in self.published
            or layer_idx not in self.keys
            or layer_idx not in self.values
        ):
            return
        key = self.keys.pop(layer_idx)
        value = self.values.pop(layer_idx)
        ready_event = None
        if key.is_cuda:
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(key.device))
        self.published.add(layer_idx)
        self.on_layer_ready(PreRopeLayerTask(layer_idx, key, value, ready_event))


class StreamingPreRopeDynamicCache(DynamicCache):
    """Normal model cache that publishes V for pairing with hooked pre-RoPE K."""

    def __init__(self, publisher: StreamingPreRopeKVPublisher):
        super().__init__()
        self.publisher = publisher

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key, value = super().update(
            key_states, value_states, layer_idx, cache_kwargs=cache_kwargs
        )
        if int(layer_idx) not in self.publisher.published:
            self.publisher.capture_value(int(layer_idx), value)
        return key, value


def _decoder_layers_and_rotary(model: Any) -> tuple[Any, Any]:
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    rotary_emb = getattr(base_model, "rotary_emb", None)
    if layers is None or rotary_emb is None:
        raise RuntimeError(
            "pre-RoPE cache alignment requires a Qwen-style model exposing "
            ".model.layers and .model.rotary_emb."
        )
    return layers, rotary_emb


def _reshape_projected_key(output: torch.Tensor, head_dim: int) -> torch.Tensor:
    if output.ndim != 3 or output.shape[-1] % head_dim:
        raise ValueError(
            f"Expected projected K [B,S,H*D], got {tuple(output.shape)} with head_dim={head_dim}."
        )
    batch_size, sequence_length, feature_dim = output.shape
    return (
        output.reshape(batch_size, sequence_length, feature_dim // head_dim, head_dim)
        .transpose(1, 2)
        .contiguous()
    )


@contextmanager
def capture_pre_rope_keys(model: Any) -> Iterator[dict[int, torch.Tensor]]:
    """Capture each Qwen attention layer's K after optional k_norm and before RoPE."""

    layers, _ = _decoder_layers_and_rotary(model)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_index: int, attention: Any):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            key = _reshape_projected_key(output, int(attention.head_dim))
            key_norm = getattr(attention, "k_norm", None)
            if key_norm is not None:
                key = key_norm(key)
            captured[layer_index] = key.detach()

        return hook

    try:
        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            hooks.append(
                attention.k_proj.register_forward_hook(make_hook(layer_index, attention))
            )
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


@contextmanager
def stream_pre_rope_keys(
    model: Any, publisher: StreamingPreRopeKVPublisher
) -> Iterator[None]:
    """Publish each pre-RoPE K immediately instead of collecting a full cache."""

    layers, _ = _decoder_layers_and_rotary(model)
    hooks = []

    def make_hook(layer_index: int, attention: Any):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            key = _reshape_projected_key(output, int(attention.head_dim))
            key_norm = getattr(attention, "k_norm", None)
            if key_norm is not None:
                key = key_norm(key)
            publisher.capture_key(layer_index, key)

        return hook

    try:
        for layer_index, layer in enumerate(layers):
            attention = layer.self_attn
            hooks.append(
                attention.k_proj.register_forward_hook(make_hook(layer_index, attention))
            )
        yield
    finally:
        for hook in hooks:
            hook.remove()


def replace_cache_keys_with_pre_rope(
    past_key_values: Any,
    captured_keys: dict[int, torch.Tensor],
) -> DynamicCache:
    """Combine captured pre-RoPE keys with values from a normal model cache."""

    if isinstance(past_key_values, DynamicCache):
        values = past_key_values.value_cache
    elif hasattr(past_key_values, "to_legacy_cache"):
        values = [value for _key, value in past_key_values.to_legacy_cache()]
    else:
        values = [value for _key, value in past_key_values]
    if len(captured_keys) != len(values):
        raise RuntimeError(
            f"Captured pre-RoPE K for {len(captured_keys)} layers, but cache has {len(values)} layers."
        )
    cache = DynamicCache()
    for layer_index, value in enumerate(values):
        cache.key_cache.append(captured_keys[layer_index])
        cache.value_cache.append(value.detach())
    return cache


def apply_receiver_compact_rope(model: Any, past_key_values: Any) -> DynamicCache:
    """Rotate receiver-layout prefix K at compact receiver positions 0..S-1."""

    if isinstance(past_key_values, DynamicCache):
        keys = past_key_values.key_cache
        values = past_key_values.value_cache
    elif hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        keys = [key for key, _value in legacy]
        values = [value for _key, value in legacy]
    else:
        legacy = tuple(past_key_values)
        keys = [key for key, _value in legacy]
        values = [value for _key, value in legacy]
    if not keys:
        raise ValueError("Cannot apply receiver RoPE to an empty cache.")

    layers, rotary_emb = _decoder_layers_and_rotary(model)
    if len(layers) != len(keys):
        raise ValueError(f"Receiver has {len(layers)} layers, but prefix has {len(keys)} layers.")
    first_key = keys[0]
    batch_size, _, sequence_length, _ = first_key.shape
    position_ids = (
        torch.arange(sequence_length, device=first_key.device, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    cos, sin = rotary_emb(first_key, position_ids)
    modeling_module = importlib.import_module(layers[0].self_attn.__class__.__module__)
    apply_rotary_pos_emb = getattr(modeling_module, "apply_rotary_pos_emb")

    rotated = DynamicCache()
    for key, value in zip(keys, values):
        _, rotated_key = apply_rotary_pos_emb(key, key, cos, sin)
        rotated.key_cache.append(rotated_key.contiguous())
        rotated.value_cache.append(value.contiguous())
    return rotated
