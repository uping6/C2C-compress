"""Trainable coefficient quantization-table selection for routed sharer KV caches.

The allocator and QAT formulation are adapted from
``/data/smy/AdaptiveJPEG-KV/src/homo_c2c_kv/training/adaptive_jpeg.py``.
This local copy deliberately has no dependency on the source research tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AdaptiveQuantTableConfig:
    enabled: bool = False
    feature_bands: int = 8
    hidden_dim: int = 128
    alpha_candidates: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0)
    initial_alpha_index: int = 0
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    anneal_steps: int = 10_000
    q_base_min: float = 1.0
    q_base_max: float = 8.0
    q_base_power: float = 1.0
    scale_side_info_bits: int = 16
    detach_allocator_features: bool = True
    rate_weight: float = 0.0
    rate_warmup_steps: int = 0


def resolve_adaptive_quant_table_config(
    config: dict[str, Any] | None,
) -> AdaptiveQuantTableConfig:
    cfg = dict(config or {})
    alpha_candidates = tuple(
        float(value)
        for value in cfg.get("alpha_candidates", (0.125, 0.25, 0.5, 1.0, 2.0))
    )
    resolved = AdaptiveQuantTableConfig(
        enabled=bool(cfg.get("enabled", False)),
        feature_bands=int(cfg.get("feature_bands", 8)),
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        alpha_candidates=alpha_candidates,
        initial_alpha_index=int(cfg.get("initial_alpha_index", 0)),
        initial_temperature=float(cfg.get("initial_temperature", 1.0)),
        final_temperature=float(cfg.get("final_temperature", 0.1)),
        anneal_steps=int(cfg.get("anneal_steps", 10_000)),
        q_base_min=float(cfg.get("q_base_min", 1.0)),
        q_base_max=float(cfg.get("q_base_max", 8.0)),
        q_base_power=float(cfg.get("q_base_power", 1.0)),
        scale_side_info_bits=int(cfg.get("scale_side_info_bits", 16)),
        detach_allocator_features=bool(cfg.get("detach_allocator_features", True)),
        rate_weight=float(cfg.get("rate_weight", 0.0)),
        rate_warmup_steps=int(cfg.get("rate_warmup_steps", 0)),
    )
    if not resolved.alpha_candidates or any(value <= 0 for value in resolved.alpha_candidates):
        raise ValueError("adaptive_quant_table.alpha_candidates must be positive.")
    if not 0 <= resolved.initial_alpha_index < len(resolved.alpha_candidates):
        raise ValueError("adaptive_quant_table.initial_alpha_index is out of range.")
    if min(resolved.feature_bands, resolved.hidden_dim) <= 0:
        raise ValueError("adaptive_quant_table feature_bands and hidden_dim must be positive.")
    if not 0 < resolved.final_temperature <= resolved.initial_temperature:
        raise ValueError(
            "adaptive_quant_table temperatures must satisfy 0 < final <= initial."
        )
    if resolved.anneal_steps < 0 or resolved.rate_warmup_steps < 0:
        raise ValueError("adaptive_quant_table anneal/warmup steps must be non-negative.")
    if resolved.q_base_min <= 0 or resolved.q_base_max < resolved.q_base_min:
        raise ValueError("adaptive_quant_table q_base range is invalid.")
    if resolved.q_base_power <= 0 or resolved.rate_weight < 0:
        raise ValueError("adaptive_quant_table q_base_power must be positive and rate_weight non-negative.")
    return resolved


@dataclass(frozen=True)
class AdaptiveQuantTableResult:
    past_key_values: tuple[tuple[Tensor, Tensor], ...]
    estimated_entropy_bits: Tensor
    estimated_payload_bits: Tensor
    entropy_bits_per_group: Tensor
    alpha: Tensor
    table_indices: Tensor
    scale: Tensor
    rounded_symbols: Tensor


def _move_axis_to_last(values: Tensor, axis: int) -> tuple[Tensor, int]:
    normalized_axis = axis if axis >= 0 else values.ndim + axis
    return values.movedim(normalized_axis, -1), normalized_axis


def _dct_ii_ortho(values: Tensor, axis: int) -> Tensor:
    """FFT-based orthonormal DCT-II suitable for long cache sequences."""

    x, original_axis = _move_axis_to_last(values, axis)
    length = int(x.shape[-1])
    if length == 0:
        return values.float()
    x = x.float().contiguous()
    original_shape = x.shape
    flattened = x.reshape(-1, length)
    reordered = torch.cat(
        (flattened[:, ::2], flattened[:, 1::2].flip(dims=(1,))), dim=1
    )
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = -torch.arange(length, device=x.device, dtype=x.dtype) * math.pi / (2 * length)
    coefficients = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    coefficients[:, :1] /= 2.0 * math.sqrt(length)
    if length > 1:
        coefficients[:, 1:] /= 2.0 * math.sqrt(length / 2.0)
    return (2.0 * coefficients).reshape(original_shape).movedim(-1, original_axis)


def _idct_iii_ortho(coefficients: Tensor, axis: int) -> Tensor:
    """Inverse of :func:`_dct_ii_ortho`."""

    x, original_axis = _move_axis_to_last(coefficients, axis)
    length = int(x.shape[-1])
    if length == 0:
        return coefficients.float()
    x = x.float().contiguous()
    original_shape = x.shape
    flattened = x.reshape(-1, length) / 2.0
    flattened[:, :1] *= 2.0 * math.sqrt(length)
    if length > 1:
        flattened[:, 1:] *= 2.0 * math.sqrt(length / 2.0)
    phase = torch.arange(length, device=x.device, dtype=x.dtype) * math.pi / (2 * length)
    imaginary = torch.cat(
        (torch.zeros_like(flattened[:, :1]), -flattened.flip(dims=(1,))[:, :-1]),
        dim=1,
    )
    real = flattened * torch.cos(phase) - imaginary * torch.sin(phase)
    imag = flattened * torch.sin(phase) + imaginary * torch.cos(phase)
    reordered = torch.fft.ifft(torch.complex(real, imag), dim=1).real
    restored = torch.empty_like(reordered)
    even_count = length - length // 2
    restored[:, ::2] = reordered[:, :even_count]
    restored[:, 1::2] = reordered.flip(dims=(1,))[:, : length // 2]
    return restored.reshape(original_shape).movedim(-1, original_axis)


class AdaptiveCoefficientQuantizer(nn.Module):
    """Select a discrete sequence-frequency quantization table per L/H/KV group."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        config: AdaptiveQuantTableConfig,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or num_kv_heads <= 0:
            raise ValueError("num_layers and num_kv_heads must be positive.")
        self.config = config
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.num_groups = self.num_layers * self.num_kv_heads * 2
        self.feature_bands = int(config.feature_bands)
        self.register_buffer(
            "alpha_candidates", torch.tensor(config.alpha_candidates, dtype=torch.float32)
        )
        self.register_buffer(
            "gumbel_temperature", torch.tensor(config.initial_temperature, dtype=torch.float32)
        )

        # Flattened cache order is [layer, K/V, head].
        layer_ids = torch.arange(self.num_layers).view(-1, 1, 1).expand(-1, 2, self.num_kv_heads)
        kv_ids = torch.arange(2).view(1, -1, 1).expand(self.num_layers, -1, self.num_kv_heads)
        head_ids = torch.arange(self.num_kv_heads).view(1, 1, -1).expand(self.num_layers, 2, -1)
        self.register_buffer("group_layer_ids", layer_ids.reshape(-1), persistent=False)
        self.register_buffer("group_head_ids", head_ids.reshape(-1), persistent=False)
        self.register_buffer("group_kv_ids", kv_ids.reshape(-1), persistent=False)

        embedding_dim = max(8, config.hidden_dim // 8)
        self.layer_embedding = nn.Embedding(self.num_layers, embedding_dim)
        self.head_embedding = nn.Embedding(self.num_kv_heads, embedding_dim)
        self.kv_embedding = nn.Embedding(2, embedding_dim)
        local_feature_dim = self.feature_bands + 3 + 3 * embedding_dim
        self.local_encoder = nn.Sequential(
            nn.Linear(local_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(self.num_groups * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(2 * config.hidden_dim, len(config.alpha_candidates))
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, -2.0)
        with torch.no_grad():
            self.alpha_head.bias[config.initial_alpha_index] = 2.0
        self.log_entropy_scales = nn.Parameter(
            torch.zeros(self.num_groups, self.feature_bands)
        )
        self.last_result: AdaptiveQuantTableResult | None = None

    def update_temperature(self, step: int) -> float:
        if self.config.anneal_steps == 0:
            progress = 1.0
        else:
            progress = min(max(int(step), 0) / self.config.anneal_steps, 1.0)
        ratio = self.config.final_temperature / self.config.initial_temperature
        temperature = self.config.initial_temperature * (ratio ** progress)
        self.gumbel_temperature.fill_(temperature)
        return float(temperature)

    def rate_weight(self, step: int) -> float:
        if self.config.rate_warmup_steps <= 0:
            return self.config.rate_weight
        return self.config.rate_weight * min(max(int(step), 0) / self.config.rate_warmup_steps, 1.0)

    def _validate_and_flatten(
        self, past_key_values: Sequence[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, int, int]:
        if len(past_key_values) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} routed source cache layers, got {len(past_key_values)}."
            )
        reference_shape = None
        layers = []
        for layer_index, (key, value) in enumerate(past_key_values):
            if key.ndim != 4 or key.shape != value.shape:
                raise ValueError(
                    f"Layer {layer_index} must contain matching [B,H,S,D] K/V tensors."
                )
            if key.shape[1] != self.num_kv_heads:
                raise ValueError(
                    f"Layer {layer_index} has {key.shape[1]} KV heads, expected {self.num_kv_heads}."
                )
            if reference_shape is None:
                reference_shape = tuple(key.shape)
            elif tuple(key.shape) != reference_shape:
                raise ValueError("Adaptive quantization requires equal KV shapes across layers.")
            layers.append(torch.stack((key, value), dim=1))
        assert reference_shape is not None
        batch, _, sequence_length, head_dim = reference_shape
        stacked = torch.stack(layers, dim=1)
        return stacked.reshape(batch, self.num_groups, sequence_length, head_dim), sequence_length, head_dim

    def _frequency_features(self, coefficients: Tensor, scale: Tensor) -> Tensor:
        batch, groups, sequence_length, _ = coefficients.shape
        power = coefficients.square().mean(dim=-1)
        pooled = F.adaptive_avg_pool1d(
            power.reshape(batch * groups, 1, sequence_length), self.feature_bands
        ).reshape(batch, groups, self.feature_bands)
        total = power.sum(dim=-1).clamp_min(1e-12)
        low_count = max(1, math.ceil(sequence_length * 0.1))
        low_ratio = power[:, :, :low_count].sum(dim=-1) / total
        probability = power / total.unsqueeze(-1)
        spectral_entropy = -(
            probability * probability.clamp_min(1e-12).log()
        ).sum(dim=-1) / math.log(max(sequence_length, 2))
        features = torch.cat(
            (
                pooled.clamp_min(1e-12).log(),
                torch.stack((scale.log(), low_ratio, spectral_entropy), dim=-1),
            ),
            dim=-1,
        )
        return features.detach() if self.config.detach_allocator_features else features

    def _select_alpha(self, coefficients: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
        batch = coefficients.shape[0]
        embeddings = torch.cat(
            (
                self.layer_embedding(self.group_layer_ids),
                self.head_embedding(self.group_head_ids),
                self.kv_embedding(self.group_kv_ids),
            ),
            dim=-1,
        ).unsqueeze(0).expand(batch, -1, -1)
        local = self.local_encoder(
            torch.cat((self._frequency_features(coefficients, scale), embeddings), dim=-1)
        )
        global_state = self.global_encoder(local.reshape(batch, -1))
        logits = self.alpha_head(
            torch.cat((local, global_state.unsqueeze(1).expand(-1, self.num_groups, -1)), dim=-1)
        )
        if self.training:
            selection = F.gumbel_softmax(
                logits, tau=float(self.gumbel_temperature.item()), hard=True, dim=-1
            )
        else:
            indices = logits.argmax(dim=-1)
            selection = F.one_hot(indices, self.alpha_candidates.numel()).to(logits.dtype)
        alpha = selection @ self.alpha_candidates.to(dtype=logits.dtype)
        return alpha, selection.argmax(dim=-1)

    def _entropy_bits(self, symbols: Tensor) -> Tensor:
        sequence_length = symbols.shape[2]
        band_index = torch.div(
            torch.arange(sequence_length, device=symbols.device) * self.feature_bands,
            sequence_length,
            rounding_mode="floor",
        ).clamp_max(self.feature_bands - 1)
        entropy_scale = F.softplus(self.log_entropy_scales) + 1e-3
        entropy_scale = entropy_scale[:, band_index].view(
            1, self.num_groups, sequence_length, 1
        )
        upper = torch.sigmoid((symbols + 0.5) / entropy_scale)
        lower = torch.sigmoid((symbols - 0.5) / entropy_scale)
        return -torch.log2((upper - lower).clamp_min(1e-9))

    def forward(
        self, past_key_values: Sequence[tuple[Tensor, Tensor]]
    ) -> AdaptiveQuantTableResult:
        flattened, sequence_length, head_dim = self._validate_and_flatten(past_key_values)
        batch = flattened.shape[0]
        coefficients = _dct_ii_ortho(flattened, axis=2)
        scale = coefficients.square().mean(dim=(-2, -1)).clamp_min(1e-12).sqrt().detach()
        alpha, table_indices = self._select_alpha(coefficients, scale)
        frequency = torch.linspace(
            0.0, 1.0, sequence_length, device=coefficients.device, dtype=coefficients.dtype
        )
        q_base = self.config.q_base_min + (
            self.config.q_base_max - self.config.q_base_min
        ) * frequency.pow(self.config.q_base_power)
        quant_step = alpha.unsqueeze(-1) * q_base.view(1, 1, -1)
        before_round = coefficients / scale.unsqueeze(-1).unsqueeze(-1)
        before_round = before_round / quant_step.unsqueeze(-1)
        rounded = torch.round(before_round)
        symbols = before_round + (rounded - before_round).detach()
        reconstructed_coefficients = (
            symbols
            * quant_step.unsqueeze(-1)
            * scale.unsqueeze(-1).unsqueeze(-1)
        )
        reconstructed = _idct_iii_ortho(reconstructed_coefficients, axis=2).to(flattened.dtype)
        cache = reconstructed.reshape(
            batch,
            self.num_layers,
            2,
            self.num_kv_heads,
            sequence_length,
            head_dim,
        )
        reconstructed_cache = tuple(
            (cache[:, layer, 0], cache[:, layer, 1])
            for layer in range(self.num_layers)
        )

        entropy_bits_per_group = self._entropy_bits(symbols).sum(dim=(-2, -1))
        estimated_entropy_bits = entropy_bits_per_group.sum()
        table_bits = batch * self.num_groups * math.ceil(
            math.log2(len(self.config.alpha_candidates))
        )
        scale_bits = batch * self.num_groups * self.config.scale_side_info_bits
        estimated_payload_bits = estimated_entropy_bits + estimated_entropy_bits.new_tensor(
            table_bits + scale_bits
        )
        result = AdaptiveQuantTableResult(
            past_key_values=reconstructed_cache,
            estimated_entropy_bits=estimated_entropy_bits,
            estimated_payload_bits=estimated_payload_bits,
            entropy_bits_per_group=entropy_bits_per_group,
            alpha=alpha.reshape(batch, self.num_layers, 2, self.num_kv_heads),
            table_indices=table_indices.reshape(batch, self.num_layers, 2, self.num_kv_heads),
            scale=scale.reshape(batch, self.num_layers, 2, self.num_kv_heads),
            rounded_symbols=rounded.reshape(
                batch,
                self.num_layers,
                2,
                self.num_kv_heads,
                sequence_length,
                head_dim,
            ),
        )
        self.last_result = result
        return result
