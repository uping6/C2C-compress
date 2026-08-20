"""LCF transport variant with learned K/V projections from a shared latent."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from rosetta.model.projector import Projector
from rosetta.utils.registry import capture_init_args, register_model


@register_model
@capture_init_args
class LCFProjectedKVProjector(Projector):
    """Encode Sharer KV jointly, then learn independent transport K/V views."""

    def __init__(
        self,
        sharer_num_kv_heads: int,
        sharer_head_dim: int,
        receiver_num_kv_heads: int,
        receiver_head_dim: int,
        shared_latent_dim: int = 128,
        key_latent_dim: int = 64,
        value_latent_dim: int = 64,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if min(shared_latent_dim, key_latent_dim, value_latent_dim) <= 0:
            raise ValueError("LCF projected-KV latent dimensions must be positive.")
        if key_latent_dim != value_latent_dim:
            raise ValueError(
                "CacheJPEG pseudo K/V currently require equal key/value latent dimensions."
            )
        self.sharer_num_kv_heads = int(sharer_num_kv_heads)
        self.sharer_head_dim = int(sharer_head_dim)
        self.receiver_num_kv_heads = int(receiver_num_kv_heads)
        self.receiver_head_dim = int(receiver_head_dim)
        self.shared_latent_dim = int(shared_latent_dim)
        self.key_latent_dim = int(key_latent_dim)
        self.value_latent_dim = int(value_latent_dim)

        source_channels = self.sharer_num_kv_heads * self.sharer_head_dim
        receiver_channels = self.receiver_num_kv_heads * self.receiver_head_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(2 * source_channels, self.shared_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(self.shared_latent_dim, 4 * self.shared_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * self.shared_latent_dim, self.shared_latent_dim, dtype=dtype),
        )
        self.key_projection = nn.Linear(
            self.shared_latent_dim, self.key_latent_dim, dtype=dtype
        )
        self.value_projection = nn.Linear(
            self.shared_latent_dim, self.value_latent_dim, dtype=dtype
        )
        self.decoder_k = nn.Sequential(
            nn.Linear(self.key_latent_dim, 4 * self.key_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * self.key_latent_dim, receiver_channels, dtype=dtype),
        )
        self.decoder_v = nn.Sequential(
            nn.Linear(self.value_latent_dim, 4 * self.value_latent_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(4 * self.value_latent_dim, receiver_channels, dtype=dtype),
        )

    def encode_shared(self, source_kv: tuple[Tensor, Tensor]) -> Tensor:
        key, value = source_kv
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("LCF projected-KV expects matching Sharer K/V [B,H,S,D].")
        batch, heads, sequence_length, head_dim = key.shape
        if (heads, head_dim) != (self.sharer_num_kv_heads, self.sharer_head_dim):
            raise ValueError("Sharer KV geometry does not match the shared LCF encoder.")
        channels = torch.cat(
            (
                key.transpose(1, 2).contiguous().reshape(batch, sequence_length, -1),
                value.transpose(1, 2).contiguous().reshape(batch, sequence_length, -1),
            ),
            dim=-1,
        )
        return self.shared_encoder(
            channels.to(dtype=self.shared_encoder[0].weight.dtype)
        )

    def project_transport(self, shared: Tensor) -> tuple[Tensor, Tensor]:
        if shared.ndim != 3 or shared.shape[-1] != self.shared_latent_dim:
            raise ValueError(
                "Expected shared latent "
                f"[B,S,{self.shared_latent_dim}], got {tuple(shared.shape)}."
            )
        return self.key_projection(shared), self.value_projection(shared)

    def encode(self, source_kv: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        return self.project_transport(self.encode_shared(source_kv))

    def decode_transport(
        self, key_latent: Tensor, value_latent: Tensor
    ) -> tuple[Tensor, Tensor]:
        if key_latent.ndim != 3 or value_latent.ndim != 3:
            raise ValueError("Projected transport K/V must have shape [B,S,C].")
        if key_latent.shape[:2] != value_latent.shape[:2]:
            raise ValueError("Projected transport K/V batch and sequence shapes must match.")
        if key_latent.shape[-1] != self.key_latent_dim:
            raise ValueError("Projected transport K has the wrong channel dimension.")
        if value_latent.shape[-1] != self.value_latent_dim:
            raise ValueError("Projected transport V has the wrong channel dimension.")
        batch, sequence_length, _ = key_latent.shape
        key_channels = self.decoder_k(
            key_latent.to(dtype=self.decoder_k[0].weight.dtype)
        )
        value_channels = self.decoder_v(
            value_latent.to(dtype=self.decoder_v[0].weight.dtype)
        )
        key = key_channels.reshape(
            batch, sequence_length, self.receiver_num_kv_heads, self.receiver_head_dim
        ).transpose(1, 2).contiguous()
        value = value_channels.reshape(
            batch, sequence_length, self.receiver_num_kv_heads, self.receiver_head_dim
        ).transpose(1, 2).contiguous()
        return key, value

    def forward(
        self,
        source_kv: tuple[Tensor, Tensor],
        target_kv: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        del target_kv
        return self.decode_transport(*self.encode(source_kv))
