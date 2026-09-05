"""Wire codec for a trained :class:`AdaptiveCoefficientQuantizer`.

The training module returns reconstructed tensors for QAT as well as the
discrete DCT symbols and their side information.  Evaluation must transmit the
discrete representation, rather than running a second fixed CacheJPEG
quantizer over the reconstructed tensors.  This module packs those symbols and
reconstructs the LCF transport K/V at the receiver boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from rosetta.cachejpeg.entropy_backends import (
    pack_adaptive_int,
    pack_dense_int16,
    pack_dense_int8,
    unpack_adaptive_int,
    unpack_dense_int16,
    unpack_dense_int8,
)
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    AdaptiveQuantTableResult,
    _idct_iii_ortho,
)


@dataclass(frozen=True)
class AdaptiveQuantizedCachePayload:
    packed_symbols: dict[str, Any]
    symbols_shape: tuple[int, ...]
    table_indices: bytes
    table_indices_shape: tuple[int, ...]
    scales: bytes
    scales_shape: tuple[int, ...]
    scales_dtype: str
    source_dtype: str


def _pack_symbols(values: np.ndarray, representation: str, backend: str) -> dict[str, Any]:
    minimum = int(values.min()) if values.size else 0
    maximum = int(values.max()) if values.size else 0
    if representation == "dense_int8":
        if minimum < -128 or maximum > 127:
            raise ValueError(
                "Adaptive quantization symbols exceed INT8; use dense_int16 or "
                f"adaptive_int. Observed range [{minimum}, {maximum}]."
            )
        return pack_dense_int8(values, backend=backend)
    if minimum < -32768 or maximum > 32767:
        raise ValueError(
            "Adaptive quantization symbols exceed INT16; observed range "
            f"[{minimum}, {maximum}]."
        )
    if representation == "adaptive_int":
        return pack_adaptive_int(values, backend=backend)
    if representation == "dense_int16":
        return pack_dense_int16(values, backend=backend)
    raise ValueError(f"Unsupported adaptive quantization representation: {representation}")


def _unpack_symbols(packed: dict[str, Any]) -> np.ndarray:
    representation = str(packed["representation"])
    if representation == "adaptive_int":
        return unpack_adaptive_int(packed)
    if representation == "dense_int8":
        return unpack_dense_int8(packed)
    if representation == "dense_int16":
        return unpack_dense_int16(packed)
    raise ValueError(f"Unsupported adaptive quantization representation: {representation}")


def encode_adaptive_quantized_cache(
    quantizer: AdaptiveCoefficientQuantizer,
    past_key_values: Sequence[tuple[Tensor, Tensor]],
    *,
    representation: str,
    backend: str,
) -> tuple[AdaptiveQuantizedCachePayload, AdaptiveQuantTableResult]:
    """Run the trained allocator and serialize its discrete wire variables."""

    with torch.no_grad():
        result = quantizer(past_key_values)
    symbols = result.rounded_symbols.detach().to(torch.int64).cpu().numpy()
    symbols_shape = tuple(int(value) for value in symbols.shape)
    batch, num_layers, key_value, num_heads, sequence_length, head_dim = symbols_shape
    wire_symbols = symbols.reshape(
        batch, num_layers * key_value * num_heads, sequence_length, head_dim
    )
    packed_symbols = _pack_symbols(wire_symbols, representation, backend)
    table_indices = result.table_indices.detach().to(torch.uint8).cpu().numpy()
    scale_numpy_dtype = np.float16 if quantizer.config.scale_side_info_bits <= 16 else np.float32
    scales = result.scale.detach().float().cpu().numpy().astype(scale_numpy_dtype)
    source_dtype = str(past_key_values[0][0].dtype)
    return (
        AdaptiveQuantizedCachePayload(
            packed_symbols=packed_symbols,
            symbols_shape=symbols_shape,
            table_indices=table_indices.tobytes(order="C"),
            table_indices_shape=tuple(int(value) for value in table_indices.shape),
            scales=scales.tobytes(order="C"),
            scales_shape=tuple(int(value) for value in scales.shape),
            scales_dtype=str(scales.dtype),
            source_dtype=source_dtype,
        ),
        result,
    )


def decode_adaptive_quantized_cache(
    payload: AdaptiveQuantizedCachePayload,
    quantizer: AdaptiveCoefficientQuantizer,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[Tensor, Tensor], ...]:
    """Reconstruct LCF transport K/V from transmitted symbols and side info."""

    wire_symbols = _unpack_symbols(payload.packed_symbols)
    expected_values = int(np.prod(payload.symbols_shape))
    if wire_symbols.size != expected_values:
        raise ValueError(
            "Adaptive payload symbol count does not match its declared shape: "
            f"decoded={wire_symbols.size}, expected={expected_values}."
        )
    symbols = torch.from_numpy(
        wire_symbols.reshape(payload.symbols_shape).copy()
    ).to(device=device, dtype=torch.float32)
    table_indices = np.frombuffer(payload.table_indices, dtype=np.uint8).reshape(
        payload.table_indices_shape
    )
    table_indices_tensor = torch.from_numpy(table_indices.copy()).to(
        device=device, dtype=torch.long
    )
    scales = np.frombuffer(payload.scales, dtype=np.dtype(payload.scales_dtype)).reshape(
        payload.scales_shape
    )
    scale_tensor = torch.from_numpy(scales.copy()).to(device=device, dtype=torch.float32)

    batch, num_layers, _, num_heads, sequence_length, head_dim = symbols.shape
    if num_layers != quantizer.num_layers or num_heads != quantizer.num_kv_heads:
        raise ValueError(
            "Adaptive payload geometry does not match the loaded quantizer: "
            f"payload layers/heads={num_layers}/{num_heads}, "
            f"quantizer={quantizer.num_layers}/{quantizer.num_kv_heads}."
        )
    alpha = quantizer.alpha_candidates.to(device=device)[table_indices_tensor]
    frequency = torch.linspace(0.0, 1.0, sequence_length, device=device)
    q_base = quantizer.config.q_base_min + (
        quantizer.config.q_base_max - quantizer.config.q_base_min
    ) * frequency.pow(quantizer.config.q_base_power)
    coefficients = (
        symbols
        * alpha.unsqueeze(-1).unsqueeze(-1)
        * q_base.view(1, 1, 1, 1, -1, 1)
        * scale_tensor.unsqueeze(-1).unsqueeze(-1)
    )
    flattened = coefficients.reshape(
        batch, num_layers * 2 * num_heads, sequence_length, head_dim
    )
    reconstructed = _idct_iii_ortho(flattened, axis=2).to(dtype=dtype)
    cache = reconstructed.reshape(
        batch, num_layers, 2, num_heads, sequence_length, head_dim
    )
    return tuple(
        (cache[:, layer_index, 0], cache[:, layer_index, 1])
        for layer_index in range(num_layers)
    )
