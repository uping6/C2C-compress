"""Direct pre-RoPE Sharer-to-Receiver KV projection for concat ablations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rosetta.model.projector import Projector
from rosetta.utils.registry import capture_init_args, register_model


@register_model
@capture_init_args
class DirectPreRopeMLPProjector(Projector):
    """Map Sharer K and V independently into Receiver cache geometry.

    This projector intentionally has no latent/wire representation.  It is an
    uncompressed local upper-bound ablation: pre-RoPE Sharer K and ordinary V
    are flattened over heads, transformed by separate MLPs, and reshaped into
    Receiver KV heads. Receiver RoPE is applied later by the concat aligner.
    """

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        hidden_dim: int = 1024,
        activation: str = "gelu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(
            sharer_num_kv_heads,
            sharer_head_dim,
            receiver_num_kv_heads,
            receiver_head_dim,
            hidden_dim,
        ) <= 0:
            raise ValueError("Direct pre-RoPE MLP dimensions must be positive.")
        activation_name = str(activation).lower()
        activation_classes = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation_name not in activation_classes:
            raise ValueError(
                "Direct pre-RoPE MLP activation must be 'gelu', 'relu', or 'silu'."
            )

        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation_name

        source_channels = self.sharer_num_kv_heads * self.sharer_head_dim
        receiver_channels = self.receiver_num_kv_heads * self.receiver_head_dim
        activation_cls = activation_classes[activation_name]
        self.key_mlp = nn.Sequential(
            nn.Linear(source_channels, self.hidden_dim, dtype=dtype),
            activation_cls(),
            nn.Linear(self.hidden_dim, receiver_channels, dtype=dtype),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(source_channels, self.hidden_dim, dtype=dtype),
            activation_cls(),
            nn.Linear(self.hidden_dim, receiver_channels, dtype=dtype),
        )

    def project(self, source_kv: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        key, value = source_kv
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError(
                "Direct pre-RoPE MLP expects matching Sharer K/V [B,H,S,D]."
            )
        batch, heads, sequence_length, head_dim = key.shape
        if (heads, head_dim) != (
            self.sharer_num_kv_heads,
            self.sharer_head_dim,
        ):
            raise ValueError("Sharer KV geometry does not match the direct MLP.")

        key_channels = key.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        value_channels = value.transpose(1, 2).contiguous().reshape(
            batch, sequence_length, -1
        )
        projection_dtype = self.key_mlp[0].weight.dtype
        projected_key = self.key_mlp(key_channels.to(dtype=projection_dtype))
        projected_value = self.value_mlp(value_channels.to(dtype=projection_dtype))
        projected_key = projected_key.reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()
        projected_value = projected_value.reshape(
            batch,
            sequence_length,
            self.receiver_num_kv_heads,
            self.receiver_head_dim,
        ).transpose(1, 2).contiguous()
        return projected_key, projected_value

    def forward(
        self,
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        del target_kv
        return self.project(source_kv)
