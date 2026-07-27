"""LCF-style joint latent bottleneck for KV-cache fusion.

The first-stage implementation in this module is intentionally a *joint*
bottleneck: it consumes both sharer and receiver caches at the receiver-side
fusion boundary.  It does not claim that the latent is produced independently
at the sharer or transmitted over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from transformers.cache_utils import DynamicCache

from rosetta.model.projector import Projector
from rosetta.utils.registry import capture_init_args, register_model


@dataclass(frozen=True)
class LatentKVBridgeConfig:
    """Configuration shared by training and evaluation entry points."""

    enabled: bool = False
    latent_dim: int = 128
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_head_scale: bool = True
    use_layer_gate: bool = True
    gate_mode: str = "sigmoid"
    gate_init: float = -4.0
    residual_scale_init: float = 0.0
    layer_mapping: str | tuple[int, ...] = "proportional"
    share_adapter_across_layers: bool = False


def resolve_latent_kv_bridge_config(config: dict[str, Any] | None) -> LatentKVBridgeConfig:
    """Parse latent bridge settings while preserving backward-compatible defaults."""

    cfg = dict(config or {})
    gate_mode = str(cfg.get("gate_mode", "sigmoid")).lower()
    if gate_mode != "sigmoid":
        raise ValueError("First-stage latent KV fusion supports only gate_mode='sigmoid'.")

    mapping_value = cfg.get("layer_mapping", "proportional")
    if isinstance(mapping_value, str):
        layer_mapping: str | tuple[int, ...] = mapping_value.lower()
        if layer_mapping not in {"proportional", "identity"}:
            raise ValueError(
                "latent_kv_bridge.layer_mapping must be 'proportional', 'identity', "
                "or an explicit list of sharer layer indices."
            )
    else:
        layer_mapping = tuple(int(value) for value in mapping_value)

    return LatentKVBridgeConfig(
        enabled=bool(cfg.get("enabled", False)),
        latent_dim=int(cfg.get("latent_dim", 128)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.0)),
        use_head_scale=bool(cfg.get("use_head_scale", True)),
        use_layer_gate=bool(cfg.get("use_layer_gate", True)),
        gate_mode=gate_mode,
        gate_init=float(cfg.get("gate_init", -4.0)),
        residual_scale_init=float(cfg.get("residual_scale_init", 0.0)),
        layer_mapping=layer_mapping,
        share_adapter_across_layers=bool(cfg.get("share_adapter_across_layers", False)),
    )


def build_proportional_layer_mapping(
    num_sharer_layers: int,
    num_receiver_layers: int,
) -> list[int]:
    """Map every receiver layer monotonically to a sharer layer."""

    if num_sharer_layers <= 0 or num_receiver_layers <= 0:
        raise ValueError("Sharer and receiver layer counts must both be positive.")
    denominator = max(num_receiver_layers - 1, 1)
    return [
        round(receiver_idx * (num_sharer_layers - 1) / denominator)
        for receiver_idx in range(num_receiver_layers)
    ]


def resolve_layer_mapping(
    mapping: str | Sequence[int],
    num_sharer_layers: int,
    num_receiver_layers: int,
) -> list[int]:
    """Resolve and validate an identity, proportional, or explicit mapping."""

    if isinstance(mapping, str):
        if mapping == "proportional":
            result = build_proportional_layer_mapping(
                num_sharer_layers, num_receiver_layers
            )
        elif mapping == "identity":
            if num_sharer_layers != num_receiver_layers:
                raise ValueError(
                    "Identity layer mapping requires equal layer counts, got "
                    f"sharer={num_sharer_layers}, receiver={num_receiver_layers}."
                )
            result = list(range(num_receiver_layers))
        else:
            raise ValueError(f"Unsupported layer mapping mode: {mapping!r}")
    else:
        result = [int(value) for value in mapping]

    if len(result) != num_receiver_layers:
        raise ValueError(
            "Layer mapping must contain one entry per receiver layer, got "
            f"{len(result)} entries for {num_receiver_layers} layers."
        )
    invalid = [idx for idx in result if idx < 0 or idx >= num_sharer_layers]
    if invalid:
        raise ValueError(
            f"Layer mapping contains out-of-range sharer indices {invalid}; "
            f"valid range is [0, {num_sharer_layers - 1}]."
        )
    return result


@register_model
@capture_init_args
class LatentKVCompressor(Projector):
    """Fuse one sharer/receiver KV layer through a joint latent bottleneck."""

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_layer_gate: bool = True,
        use_head_scale: bool = True,
        init_residual_scale: float = 0.0,
        gate_init: float = -4.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        dimensions = {
            "sharer_num_kv_heads": sharer_num_kv_heads,
            "sharer_head_dim": sharer_head_dim,
            "receiver_num_kv_heads": receiver_num_kv_heads,
            "receiver_head_dim": receiver_head_dim,
            "latent_dim": latent_dim,
        }
        invalid = {name: value for name, value in dimensions.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"All latent KV dimensions must be positive, got {invalid}.")
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even for K/V splitting, got {latent_dim}.")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.latent_dim = int(latent_dim)
        self.mlp_ratio = float(mlp_ratio)
        self.use_layer_gate = bool(use_layer_gate)
        self.use_head_scale = bool(use_head_scale)

        sharer_width = self.sharer_num_kv_heads * self.sharer_head_dim
        receiver_width = self.receiver_num_kv_heads * self.receiver_head_dim
        self.joint_input_dim = 2 * sharer_width + 2 * receiver_width
        hidden_dim = max(1, int(self.latent_dim * self.mlp_ratio))
        half_latent_dim = self.latent_dim // 2

        self.down_proj = nn.Linear(
            self.joint_input_dim, self.latent_dim, dtype=dtype
        )
        self.latent_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim, dtype=dtype),
        )
        self.key_up_proj = nn.Linear(
            half_latent_dim, receiver_width, dtype=dtype
        )
        self.value_up_proj = nn.Linear(
            half_latent_dim, receiver_width, dtype=dtype
        )

        self.key_head_scale = nn.Parameter(
            torch.full(
                (self.receiver_num_kv_heads,),
                float(init_residual_scale),
                dtype=dtype,
            )
        )
        self.value_head_scale = nn.Parameter(
            torch.full(
                (self.receiver_num_kv_heads,),
                float(init_residual_scale),
                dtype=dtype,
            )
        )
        self.key_gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=dtype))
        self.last_stats: dict[str, Any] | None = None

    @staticmethod
    def _describe_shapes(tensors: Sequence[Tensor]) -> list[tuple[int, ...]]:
        return [tuple(tensor.shape) for tensor in tensors]

    def _validate_inputs(
        self,
        sharer_key: Tensor,
        sharer_value: Tensor,
        receiver_key: Tensor,
        receiver_value: Tensor,
    ) -> tuple[int, int]:
        tensors = (sharer_key, sharer_value, receiver_key, receiver_value)
        if any(tensor.ndim != 4 for tensor in tensors):
            raise ValueError(
                "LatentKVCompressor expects [B,H,T,D] tensors, got shapes "
                f"{self._describe_shapes(tensors)}."
            )
        if sharer_key.shape != sharer_value.shape:
            raise ValueError(
                "Sharer K/V shapes must match, got "
                f"K={tuple(sharer_key.shape)}, V={tuple(sharer_value.shape)}."
            )
        if receiver_key.shape != receiver_value.shape:
            raise ValueError(
                "Receiver K/V shapes must match, got "
                f"K={tuple(receiver_key.shape)}, V={tuple(receiver_value.shape)}."
            )
        batch, sharer_heads, sharer_length, sharer_dim = sharer_key.shape
        receiver_batch, receiver_heads, receiver_length, receiver_dim = receiver_key.shape
        expected = (
            self.sharer_num_kv_heads,
            self.sharer_head_dim,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        )
        actual = (sharer_heads, sharer_dim, receiver_heads, receiver_dim)
        if actual != expected:
            raise ValueError(
                "KV head dimensions do not match the module configuration: "
                f"actual (Hs,Ds,Hr,Dr)={actual}, expected={expected}; shapes="
                f"{self._describe_shapes(tensors)}."
            )
        if batch != receiver_batch or sharer_length != receiver_length:
            raise ValueError(
                "Joint latent fusion requires equal batch and sequence lengths, got "
                f"sharer={tuple(sharer_key.shape)}, receiver={tuple(receiver_key.shape)}."
            )
        devices = {tensor.device for tensor in tensors}
        dtypes = {tensor.dtype for tensor in tensors}
        if len(devices) != 1 or len(dtypes) != 1:
            raise ValueError(
                "All KV tensors must share a device and dtype, got "
                f"devices={devices}, dtypes={dtypes}."
            )
        parameter = self.down_proj.weight
        if parameter.device != sharer_key.device or parameter.dtype != sharer_key.dtype:
            raise ValueError(
                "LatentKVCompressor parameters must match cache device/dtype, got "
                f"module=({parameter.device}, {parameter.dtype}), "
                f"cache=({sharer_key.device}, {sharer_key.dtype})."
            )
        return batch, sharer_length

    def forward(
        self,
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        sharer_key, sharer_value = source_kv
        receiver_key, receiver_value = target_kv
        batch, sequence_length = self._validate_inputs(
            sharer_key, sharer_value, receiver_key, receiver_value
        )

        # [B,H,T,D] -> [B,T,H*D]. contiguous() is required after transpose.
        sharer_key_flat = sharer_key.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        sharer_value_flat = sharer_value.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        receiver_key_flat = receiver_key.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        receiver_value_flat = receiver_value.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        joint = torch.cat(
            [
                sharer_key_flat,
                sharer_value_flat,
                receiver_key_flat,
                receiver_value_flat,
            ],
            dim=-1,
        )

        latent = self.down_proj(joint)
        latent = latent + self.latent_mlp(latent)
        key_latent, value_latent = latent.chunk(2, dim=-1)

        # [B,T,Hr*Dr] -> [B,Hr,T,Dr].
        key_delta = self.key_up_proj(key_latent).reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()
        value_delta = self.value_up_proj(value_latent).reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()

        if self.use_head_scale:
            key_scale = torch.tanh(self.key_head_scale).reshape(1, -1, 1, 1)
            value_scale = torch.tanh(self.value_head_scale).reshape(1, -1, 1, 1)
            key_delta = key_delta * key_scale
            value_delta = value_delta * value_scale

        key_gate = (
            torch.sigmoid(self.key_gate_logit)
            if self.use_layer_gate
            else self.key_gate_logit.new_tensor(1.0)
        )
        value_gate = (
            torch.sigmoid(self.value_gate_logit)
            if self.use_layer_gate
            else self.value_gate_logit.new_tensor(1.0)
        )
        fused_key = receiver_key + key_gate * key_delta
        fused_value = receiver_value + value_gate * value_delta

        with torch.no_grad():
            receiver_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(receiver_key.float()),
                        torch.linalg.vector_norm(receiver_value.float()),
                    ]
                )
            )
            residual_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm((key_gate * key_delta).float()),
                        torch.linalg.vector_norm((value_gate * value_delta).float()),
                    ]
                )
            )
            self.last_stats = {
                "latent_dim": self.latent_dim,
                "joint_input_dim": self.joint_input_dim,
                "key_gate": float(key_gate.detach().float().cpu()),
                "value_gate": float(value_gate.detach().float().cpu()),
                "key_head_scale_mean": float(
                    torch.tanh(self.key_head_scale).float().mean().cpu()
                ),
                "key_head_scale_var": float(
                    torch.tanh(self.key_head_scale).float().var(unbiased=False).cpu()
                ),
                "value_head_scale_mean": float(
                    torch.tanh(self.value_head_scale).float().mean().cpu()
                ),
                "value_head_scale_var": float(
                    torch.tanh(self.value_head_scale).float().var(unbiased=False).cpu()
                ),
                "residual_norm": float(residual_norm.cpu()),
                "receiver_cache_norm": float(receiver_norm.cpu()),
                "residual_receiver_ratio": float(
                    (residual_norm / receiver_norm.clamp_min(1e-12)).cpu()
                ),
            }
        return fused_key, fused_value


class CacheAdapter:
    """Convert supported cache containers without mutating the input object."""

    @staticmethod
    def to_legacy(cache: Any) -> tuple[tuple[Tensor, Tensor], ...]:
        if isinstance(cache, DynamicCache):
            return tuple(zip(cache.key_cache, cache.value_cache))
        if hasattr(cache, "to_legacy_cache"):
            return tuple(cache.to_legacy_cache())
        if isinstance(cache, (tuple, list)):
            return tuple((layer[0], layer[1]) for layer in cache)
        raise TypeError(f"Unsupported KV cache container: {type(cache)!r}")

    @staticmethod
    def restore(
        layers: Sequence[tuple[Tensor, Tensor]],
        template: Any,
    ) -> Any:
        cloned_layers = tuple((key.clone(), value.clone()) for key, value in layers)
        if isinstance(template, DynamicCache):
            restored = DynamicCache()
            for key, value in cloned_layers:
                restored.key_cache.append(key)
                restored.value_cache.append(value)
            return restored
        if isinstance(template, tuple):
            return cloned_layers
        if isinstance(template, list):
            return list(cloned_layers)
        from_legacy = getattr(type(template), "from_legacy_cache", None)
        if callable(from_legacy):
            return from_legacy(cloned_layers)
        raise TypeError(
            f"Cannot restore unsupported KV cache container: {type(template)!r}"
        )


class LatentKVBridge(nn.Module):
    """Apply one joint latent adapter per receiver cache layer."""

    def __init__(
        self,
        num_sharer_layers: int,
        num_receiver_layers: int,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        layer_mapping: str | Sequence[int] = "proportional",
        latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_layer_gate: bool = True,
        use_head_scale: bool = True,
        init_residual_scale: float = 0.0,
        gate_init: float = -4.0,
        share_adapter_across_layers: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.num_sharer_layers = int(num_sharer_layers)
        self.num_receiver_layers = int(num_receiver_layers)
        self.layer_mapping = resolve_layer_mapping(
            layer_mapping, self.num_sharer_layers, self.num_receiver_layers
        )
        self.share_adapter_across_layers = bool(share_adapter_across_layers)

        adapter_kwargs = dict(
            sharer_num_kv_heads=sharer_num_kv_heads,
            sharer_head_dim=sharer_head_dim,
            receiver_num_kv_heads=receiver_num_kv_heads,
            receiver_head_dim=receiver_head_dim,
            latent_dim=latent_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_layer_gate=use_layer_gate,
            use_head_scale=use_head_scale,
            init_residual_scale=init_residual_scale,
            gate_init=gate_init,
            dtype=dtype,
        )
        adapter_count = 1 if self.share_adapter_across_layers else self.num_receiver_layers
        self.adapters = nn.ModuleList(
            [LatentKVCompressor(**adapter_kwargs) for _ in range(adapter_count)]
        )
        self.last_stats: dict[str, Any] | None = None

    def forward(self, sharer_past_key_values: Any, receiver_past_key_values: Any) -> Any:
        sharer_layers = CacheAdapter.to_legacy(sharer_past_key_values)
        receiver_layers = CacheAdapter.to_legacy(receiver_past_key_values)
        if len(sharer_layers) != self.num_sharer_layers:
            raise ValueError(
                f"Expected {self.num_sharer_layers} sharer cache layers, got {len(sharer_layers)}."
            )
        if len(receiver_layers) != self.num_receiver_layers:
            raise ValueError(
                f"Expected {self.num_receiver_layers} receiver cache layers, got {len(receiver_layers)}."
            )

        fused_layers: list[tuple[Tensor, Tensor]] = []
        layer_stats: list[dict[str, Any]] = []
        for receiver_idx, sharer_idx in enumerate(self.layer_mapping):
            adapter_idx = 0 if self.share_adapter_across_layers else receiver_idx
            adapter = self.adapters[adapter_idx]
            fused_layers.append(
                adapter(sharer_layers[sharer_idx], receiver_layers[receiver_idx])
            )
            layer_stats.append(
                {
                    "receiver_layer": receiver_idx,
                    "sharer_layer": sharer_idx,
                    **dict(adapter.last_stats or {}),
                }
            )

        sharer_elements = sum(
            key.numel() + value.numel() for key, value in sharer_layers
        )
        latent_elements = sum(
            sharer_layers[sharer_idx][0].shape[0]
            * sharer_layers[sharer_idx][0].shape[2]
            * self.adapters[0].latent_dim
            for sharer_idx in self.layer_mapping
        )
        self.last_stats = {
            "fusion_type": "latent_kv_joint",
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            # This is a representational ratio only in joint mode, not wire bytes.
            "raw_sharer_elements": int(sharer_elements),
            "latent_elements": int(latent_elements),
            "element_compression_ratio": (
                float(sharer_elements / latent_elements) if latent_elements else 0.0
            ),
            "layers": layer_stats,
        }
        return CacheAdapter.restore(fused_layers, receiver_past_key_values)


class SharerKVEncoder(nn.Module):
    """Encode one sharer K/V layer without access to receiver state."""

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(sharer_num_kv_heads, sharer_head_dim, latent_dim) <= 0:
            raise ValueError("Sharer dimensions and latent_dim must be positive.")
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even, got {latent_dim}.")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.latent_dim = int(latent_dim)
        self.input_dim = 2 * self.sharer_num_kv_heads * self.sharer_head_dim
        hidden_dim = max(1, int(self.latent_dim * mlp_ratio))
        self.down_proj = nn.Linear(self.input_dim, self.latent_dim, dtype=dtype)
        self.latent_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim, dtype=dtype),
        )

    def forward(self, sharer_key: Tensor, sharer_value: Tensor) -> Tensor:
        if sharer_key.ndim != 4 or sharer_value.ndim != 4:
            raise ValueError(
                "SharerKVEncoder expects [B,H,T,D], got "
                f"K={tuple(sharer_key.shape)}, V={tuple(sharer_value.shape)}."
            )
        if sharer_key.shape != sharer_value.shape:
            raise ValueError(
                "Sharer K/V shapes must match, got "
                f"K={tuple(sharer_key.shape)}, V={tuple(sharer_value.shape)}."
            )
        batch, heads, sequence_length, head_dim = sharer_key.shape
        if (heads, head_dim) != (
            self.sharer_num_kv_heads,
            self.sharer_head_dim,
        ):
            raise ValueError(
                "Sharer KV dimensions do not match encoder configuration: "
                f"actual={(heads, head_dim)}, expected="
                f"{(self.sharer_num_kv_heads, self.sharer_head_dim)}."
            )
        if sharer_key.device != sharer_value.device or sharer_key.dtype != sharer_value.dtype:
            raise ValueError("Sharer K/V must share a device and dtype.")
        parameter = self.down_proj.weight
        if parameter.device != sharer_key.device or parameter.dtype != sharer_key.dtype:
            raise ValueError(
                "SharerKVEncoder parameters must match cache device/dtype, got "
                f"module=({parameter.device}, {parameter.dtype}), "
                f"cache=({sharer_key.device}, {sharer_key.dtype})."
            )

        key_flat = sharer_key.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        value_flat = sharer_value.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        latent = self.down_proj(torch.cat([key_flat, value_flat], dim=-1))
        return latent + self.latent_mlp(latent)


class ReceiverKVDecoder(nn.Module):
    """Decode a sharer latent conditioned only on receiver K/V state."""

    def __init__(
        self,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_layer_gate: bool = True,
        use_head_scale: bool = True,
        init_residual_scale: float = 0.0,
        gate_init: float = -4.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(receiver_num_kv_heads, receiver_head_dim, latent_dim) <= 0:
            raise ValueError("Receiver dimensions and latent_dim must be positive.")
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even, got {latent_dim}.")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.latent_dim = int(latent_dim)
        self.use_layer_gate = bool(use_layer_gate)
        self.use_head_scale = bool(use_head_scale)
        receiver_width = self.receiver_num_kv_heads * self.receiver_head_dim
        receiver_input_dim = 2 * receiver_width
        hidden_dim = max(1, int(self.latent_dim * mlp_ratio))
        half_latent_dim = self.latent_dim // 2

        self.receiver_condition_proj = nn.Linear(
            receiver_input_dim, self.latent_dim, dtype=dtype
        )
        self.fusion_proj = nn.Linear(
            2 * self.latent_dim, self.latent_dim, dtype=dtype
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim, dtype=dtype),
        )
        self.key_up_proj = nn.Linear(half_latent_dim, receiver_width, dtype=dtype)
        self.value_up_proj = nn.Linear(half_latent_dim, receiver_width, dtype=dtype)
        self.key_head_scale = nn.Parameter(
            torch.full(
                (self.receiver_num_kv_heads,),
                float(init_residual_scale),
                dtype=dtype,
            )
        )
        self.value_head_scale = nn.Parameter(
            torch.full(
                (self.receiver_num_kv_heads,),
                float(init_residual_scale),
                dtype=dtype,
            )
        )
        self.key_gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=dtype))
        self.value_gate_logit = nn.Parameter(torch.tensor(float(gate_init), dtype=dtype))
        self.last_stats: dict[str, Any] | None = None

    def forward(
        self,
        latent: Tensor,
        receiver_key: Tensor,
        receiver_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if latent.ndim != 3:
            raise ValueError(
                f"ReceiverKVDecoder expects latent [B,T,L], got {tuple(latent.shape)}."
            )
        if receiver_key.ndim != 4 or receiver_value.ndim != 4:
            raise ValueError(
                "ReceiverKVDecoder expects K/V [B,H,T,D], got "
                f"K={tuple(receiver_key.shape)}, V={tuple(receiver_value.shape)}."
            )
        if receiver_key.shape != receiver_value.shape:
            raise ValueError(
                "Receiver K/V shapes must match, got "
                f"K={tuple(receiver_key.shape)}, V={tuple(receiver_value.shape)}."
            )
        batch, heads, sequence_length, head_dim = receiver_key.shape
        if tuple(latent.shape) != (batch, sequence_length, self.latent_dim):
            raise ValueError(
                "Latent batch/sequence/dimension must match receiver cache, got "
                f"latent={tuple(latent.shape)}, receiver={tuple(receiver_key.shape)}, "
                f"expected latent_dim={self.latent_dim}."
            )
        if (heads, head_dim) != (
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ):
            raise ValueError(
                "Receiver KV dimensions do not match decoder configuration: "
                f"actual={(heads, head_dim)}, expected="
                f"{(self.receiver_num_kv_heads, self.receiver_head_dim)}."
            )
        tensors = (latent, receiver_key, receiver_value)
        if len({tensor.device for tensor in tensors}) != 1 or len(
            {tensor.dtype for tensor in tensors}
        ) != 1:
            raise ValueError("Latent and receiver K/V must share a device and dtype.")
        parameter = self.receiver_condition_proj.weight
        if parameter.device != receiver_key.device or parameter.dtype != receiver_key.dtype:
            raise ValueError(
                "ReceiverKVDecoder parameters must match cache device/dtype, got "
                f"module=({parameter.device}, {parameter.dtype}), "
                f"cache=({receiver_key.device}, {receiver_key.dtype})."
            )

        key_flat = receiver_key.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        value_flat = receiver_value.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        receiver_condition = self.receiver_condition_proj(
            torch.cat([key_flat, value_flat], dim=-1)
        )
        fused_latent = self.fusion_proj(
            torch.cat([latent, receiver_condition], dim=-1)
        )
        fused_latent = fused_latent + self.fusion_mlp(fused_latent)
        key_latent, value_latent = fused_latent.chunk(2, dim=-1)

        key_delta = self.key_up_proj(key_latent).reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()
        value_delta = self.value_up_proj(value_latent).reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()

        if self.use_head_scale:
            key_delta = key_delta * torch.tanh(self.key_head_scale).reshape(
                1, -1, 1, 1
            )
            value_delta = value_delta * torch.tanh(self.value_head_scale).reshape(
                1, -1, 1, 1
            )
        key_gate = (
            torch.sigmoid(self.key_gate_logit)
            if self.use_layer_gate
            else self.key_gate_logit.new_tensor(1.0)
        )
        value_gate = (
            torch.sigmoid(self.value_gate_logit)
            if self.use_layer_gate
            else self.value_gate_logit.new_tensor(1.0)
        )
        key_residual = key_gate * key_delta
        value_residual = value_gate * value_delta
        fused_key = receiver_key + key_residual
        fused_value = receiver_value + value_residual

        with torch.no_grad():
            receiver_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(receiver_key.float()),
                        torch.linalg.vector_norm(receiver_value.float()),
                    ]
                )
            )
            residual_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        torch.linalg.vector_norm(key_residual.float()),
                        torch.linalg.vector_norm(value_residual.float()),
                    ]
                )
            )
            self.last_stats = {
                "latent_dim": self.latent_dim,
                "key_gate": float(key_gate.detach().float().cpu()),
                "value_gate": float(value_gate.detach().float().cpu()),
                "key_head_scale_mean": float(
                    torch.tanh(self.key_head_scale).float().mean().cpu()
                ),
                "key_head_scale_var": float(
                    torch.tanh(self.key_head_scale).float().var(unbiased=False).cpu()
                ),
                "value_head_scale_mean": float(
                    torch.tanh(self.value_head_scale).float().mean().cpu()
                ),
                "value_head_scale_var": float(
                    torch.tanh(self.value_head_scale).float().var(unbiased=False).cpu()
                ),
                "residual_norm": float(residual_norm.cpu()),
                "receiver_cache_norm": float(receiver_norm.cpu()),
                "residual_receiver_ratio": float(
                    (residual_norm / receiver_norm.clamp_min(1e-12)).cpu()
                ),
            }
        return fused_key, fused_value


@register_model
@capture_init_args
class SplitLatentKVProjector(Projector):
    """Trainable pair with an explicit sharer encoder/receiver decoder boundary."""

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        latent_dim: int = 128,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_layer_gate: bool = True,
        use_head_scale: bool = True,
        init_residual_scale: float = 0.0,
        gate_init: float = -4.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.encoder = SharerKVEncoder(
            sharer_num_kv_heads=sharer_num_kv_heads,
            sharer_head_dim=sharer_head_dim,
            latent_dim=latent_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            dtype=dtype,
        )
        self.decoder = ReceiverKVDecoder(
            receiver_num_kv_heads=receiver_num_kv_heads,
            receiver_head_dim=receiver_head_dim,
            latent_dim=latent_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_layer_gate=use_layer_gate,
            use_head_scale=use_head_scale,
            init_residual_scale=init_residual_scale,
            gate_init=gate_init,
            dtype=dtype,
        )
        self.last_stats: dict[str, Any] | None = None

    def encode(self, source_kv: tuple[Tensor, Tensor]) -> Tensor:
        """Sharer-side API; no receiver tensor is accepted or reachable."""

        return self.encoder(source_kv[0], source_kv[1])

    def decode(
        self,
        latent: Tensor,
        target_kv: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Receiver-side API producing residual-fused receiver K/V."""

        result = self.decoder(latent, target_kv[0], target_kv[1])
        self.last_stats = dict(self.decoder.last_stats or {})
        return result

    def forward(
        self,
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        # End-to-end training convenience. In deployed inference encode/decode
        # are called on opposite sides of the transport boundary.
        return self.decode(self.encode(source_kv), target_kv)


@dataclass
class LatentKVLayerPayload:
    receiver_layer: int
    sharer_layer: int
    projector_idx: int
    latent: Tensor


@dataclass
class LatentKVPayload:
    """First unquantized wire payload for split latent communication."""

    layers: list[LatentKVLayerPayload]
    latent_dim: int
    sequence_length: int
    source_dtype: str
    quantized: bool = False


@dataclass
class CacheJPEGLatentKVPayload:
    """Wire payload containing JPEG-compressed split latents and layer metadata."""

    encoded_payload: Any
    layers: list[tuple[int, int, int]]
    latent_dim: int
    sequence_length: int
    source_dtype: str
    entropy_backend: str
    codec: str = "cachejpeg"


def latent_payload_to_pseudo_kv_cache(
    payload: LatentKVPayload,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Represent [B,T,L] latents as K/V-like [B,1,T,L/2] codec inputs."""

    if payload.latent_dim % 2 != 0:
        raise ValueError(
            f"CacheJPEG latent compression requires even latent_dim, got {payload.latent_dim}."
        )
    pseudo_layers = []
    for layer in payload.layers:
        if layer.latent.ndim != 3 or layer.latent.shape[-1] != payload.latent_dim:
            raise ValueError(
                "Latent payload shape does not match metadata: "
                f"shape={tuple(layer.latent.shape)}, latent_dim={payload.latent_dim}."
            )
        key_latent, value_latent = layer.latent.chunk(2, dim=-1)
        pseudo_layers.append(
            (
                key_latent.unsqueeze(1).contiguous(),
                value_latent.unsqueeze(1).contiguous(),
            )
        )
    return tuple(pseudo_layers)


def pseudo_kv_cache_to_latent_payload(
    pseudo_cache: Sequence[tuple[Tensor, Tensor]],
    compressed_payload: CacheJPEGLatentKVPayload,
) -> LatentKVPayload:
    """Restore decoder-ready latents after CacheJPEG reconstruction."""

    if len(pseudo_cache) != len(compressed_payload.layers):
        raise ValueError(
            "Decoded CacheJPEG layer count does not match latent metadata: "
            f"decoded={len(pseudo_cache)}, metadata={len(compressed_payload.layers)}."
        )
    layers: list[LatentKVLayerPayload] = []
    for (key_latent, value_latent), metadata in zip(
        pseudo_cache, compressed_payload.layers
    ):
        if key_latent.ndim != 4 or value_latent.ndim != 4:
            raise ValueError("Decoded pseudo K/V tensors must use [B,H,T,D] layout.")
        if key_latent.shape != value_latent.shape or key_latent.shape[1] != 1:
            raise ValueError(
                "Decoded pseudo K/V must have matching shapes and one head, got "
                f"K={tuple(key_latent.shape)}, V={tuple(value_latent.shape)}."
            )
        receiver_layer, sharer_layer, projector_idx = metadata
        latent = torch.cat(
            [key_latent.squeeze(1), value_latent.squeeze(1)], dim=-1
        ).contiguous()
        if latent.shape[-1] != compressed_payload.latent_dim:
            raise ValueError(
                "Decoded latent dimension does not match metadata: "
                f"decoded={latent.shape[-1]}, expected={compressed_payload.latent_dim}."
            )
        layers.append(
            LatentKVLayerPayload(
                receiver_layer=int(receiver_layer),
                sharer_layer=int(sharer_layer),
                projector_idx=int(projector_idx),
                latent=latent,
            )
        )
    return LatentKVPayload(
        layers=layers,
        latent_dim=compressed_payload.latent_dim,
        sequence_length=compressed_payload.sequence_length,
        source_dtype=compressed_payload.source_dtype,
        # The wire representation was quantized, but these tensors have already
        # been reconstructed and are ready for the receiver decoder.
        quantized=False,
    )
