from __future__ import annotations

from typing import Any

import numpy as np
import torch

from homo_c2c_kv.codec.base import EncodedPayload
from homo_c2c_kv.codec.cachejpeg.codec import CacheJPEGCodec, _component_bytes
from homo_c2c_kv.codec.cachejpeg.config import CacheJPEGConfig, resolve_cachejpeg_config
from homo_c2c_kv.codec.cachejpeg.quant_table import (
    build_frequency_table,
    frequency_band_slice,
    layer_group,
)

from rosetta.cachejpeg.entropy_backends import (
    pack_adaptive_int,
    pack_dense_int16,
    pack_dense_int8,
    unpack_adaptive_int,
    unpack_dense_int16,
    unpack_dense_int8,
)


def _move_axis_to_last(values: torch.Tensor, axis: int) -> tuple[torch.Tensor, int]:
    normalized_axis = axis if axis >= 0 else values.ndim + axis
    return values.movedim(normalized_axis, -1), normalized_axis


def dct_ii_ortho(values: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """GPU-compatible DCT-II matching scipy.fft.dct(..., norm='ortho')."""
    x, original_axis = _move_axis_to_last(values, axis)
    length = int(x.shape[-1])
    if length == 0:
        return values.to(dtype=torch.float32)
    x = x.to(dtype=torch.float32).contiguous()
    original_shape = x.shape
    flattened = x.reshape(-1, length)
    reordered = torch.cat((flattened[:, ::2], flattened[:, 1::2].flip(dims=(1,))), dim=1)
    spectrum = torch.fft.fft(reordered, dim=1)
    phase = -torch.arange(length, device=x.device, dtype=torch.float32) * torch.pi / (2.0 * length)
    coefficients = spectrum.real * torch.cos(phase) - spectrum.imag * torch.sin(phase)
    coefficients[:, :1] /= 2.0 * np.sqrt(length)
    if length > 1:
        coefficients[:, 1:] /= 2.0 * np.sqrt(length / 2.0)
    coefficients = (2.0 * coefficients).reshape(original_shape)
    return coefficients.movedim(-1, original_axis)


def idct_iii_ortho(coefficients: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Inverse of dct_ii_ortho, matching scipy.fft.idct(..., norm='ortho')."""
    x, original_axis = _move_axis_to_last(coefficients, axis)
    length = int(x.shape[-1])
    if length == 0:
        return coefficients.to(dtype=torch.float32)
    x = x.to(dtype=torch.float32).contiguous()
    original_shape = x.shape
    flattened = x.reshape(-1, length) / 2.0
    flattened[:, :1] *= 2.0 * np.sqrt(length)
    if length > 1:
        flattened[:, 1:] *= 2.0 * np.sqrt(length / 2.0)

    phase = torch.arange(length, device=x.device, dtype=torch.float32) * torch.pi / (2.0 * length)
    real = flattened
    imag = torch.cat((torch.zeros_like(flattened[:, :1]), -flattened.flip(dims=(1,))[:, :-1]), dim=1)
    spectrum_real = real * torch.cos(phase) - imag * torch.sin(phase)
    spectrum_imag = real * torch.sin(phase) + imag * torch.cos(phase)
    reordered = torch.fft.ifft(torch.complex(spectrum_real, spectrum_imag), dim=1).real
    restored = torch.empty_like(reordered)
    even_count = length - length // 2
    restored[:, ::2] = reordered[:, :even_count]
    restored[:, 1::2] = reordered.flip(dims=(1,))[:, : length // 2]
    return restored.reshape(original_shape).movedim(-1, original_axis)


def _anchor_indices(seq_len: int, cfg: CacheJPEGConfig, device: torch.device) -> torch.Tensor:
    indices: set[int] = set(range(min(seq_len, max(0, int(cfg.anchors.sink_count)))))
    recent_count = min(seq_len, max(0, int(cfg.anchors.recent_count)))
    if recent_count:
        indices.update(range(seq_len - recent_count, seq_len))
    if cfg.anchors.preserve_options:
        indices.update(int(index) for index in cfg.anchors.option_token_indices if 0 <= int(index) < seq_len)
    return torch.tensor(sorted(indices), device=device, dtype=torch.long)


def _body_indices(seq_len: int, anchor_indices: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(seq_len, device=anchor_indices.device, dtype=torch.bool)
    if anchor_indices.numel():
        mask[anchor_indices] = False
    return torch.nonzero(mask, as_tuple=False).flatten()


def _forward_dct(values: torch.Tensor, cfg: CacheJPEGConfig) -> tuple[torch.Tensor, dict[str, Any]]:
    body_len = int(values.shape[2])
    if body_len == 0:
        return values.float(), {
            "mode": cfg.block.mode,
            "body_len": 0,
            "padded_len": 0,
            "pad_len": 0,
            "num_blocks": 0,
            "freq_len": 0,
        }
    if cfg.block.mode == "global":
        return dct_ii_ortho(values, axis=2), {
            "mode": "global",
            "body_len": body_len,
            "padded_len": body_len,
            "pad_len": 0,
            "num_blocks": 1,
            "freq_len": body_len,
        }
    if cfg.block.mode != "fixed":
        raise ValueError(f"Unsupported CacheJPEG block mode: {cfg.block.mode}")
    block_size = max(1, int(cfg.block.size))
    num_blocks = int(np.ceil(body_len / block_size))
    padded_len = num_blocks * block_size
    pad_len = padded_len - body_len
    if pad_len:
        values = torch.nn.functional.pad(values, (0, 0, 0, pad_len))
    blocked = values.reshape(values.shape[0], values.shape[1], num_blocks, block_size, values.shape[3])
    return dct_ii_ortho(blocked, axis=3), {
        "mode": "fixed",
        "body_len": body_len,
        "padded_len": padded_len,
        "pad_len": pad_len,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "freq_len": block_size,
    }


def _inverse_dct(coefficients: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    body_len = int(transform["body_len"])
    if body_len == 0:
        return coefficients.float()
    if transform["mode"] == "global":
        return idct_iii_ortho(coefficients, axis=2)
    restored = idct_iii_ortho(coefficients, axis=3)
    restored = restored.reshape(restored.shape[0], restored.shape[1], int(transform["padded_len"]), restored.shape[4])
    return restored[:, :, :body_len, :].contiguous()


def _table_view(table: torch.Tensor, transform: dict[str, Any]) -> tuple[int, ...]:
    if transform["mode"] == "fixed":
        return (1, 1, 1, int(table.numel()), 1)
    return (1, 1, int(table.numel()), 1)


def _frequency_density(values: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    if transform["mode"] == "fixed":
        return (values != 0).float().mean(dim=(0, 1, 2, 4))
    return (values != 0).float().mean(dim=(0, 1, 3))


def _apply_zero_tail(values: torch.Tensor, cfg: CacheJPEGConfig, transform: dict[str, Any]) -> int:
    freq_len = int(transform["freq_len"])
    if freq_len <= 0 or cfg.zero_tail.mode in {"none", "off", "disabled"}:
        return freq_len
    min_keep = min(freq_len, max(1, int(cfg.zero_tail.min_keep)))
    if cfg.zero_tail.mode in {"fixed", "fixed_post"}:
        keep = max(min_keep, int(round(freq_len * (1.0 - cfg.zero_tail.ratio))))
    elif cfg.zero_tail.mode == "adaptive_post":
        active = torch.nonzero(
            _frequency_density(values, transform) > cfg.zero_tail.density_threshold,
            as_tuple=False,
        ).flatten()
        keep = int(active[-1].item() + 1) if active.numel() else min_keep
        keep = max(keep, max(min_keep, int(round(freq_len * (1.0 - cfg.zero_tail.ratio)))))
    else:
        raise ValueError(f"Unsupported CacheJPEG zero_tail mode: {cfg.zero_tail.mode}")
    keep = min(freq_len, max(min_keep, keep))
    if keep < freq_len:
        if transform["mode"] == "fixed":
            values[:, :, :, keep:, :] = 0
        else:
            values[:, :, keep:, :] = 0
    return keep


def _quantize(
    coefficients: torch.Tensor,
    table: torch.Tensor,
    cfg: CacheJPEGConfig,
    transform: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch_dtype = torch.int8 if cfg.entropy.representation == "dense_int8" else torch.int16
    dtype_info = torch.iinfo(torch_dtype)
    if coefficients.numel() == 0:
        return coefficients.to(torch_dtype), {
            "nonzero_ratio": 0.0,
            "clip_ratio": 0.0,
            "abs_max_preclip": 0.0,
            "abs_p99_preclip": 0.0,
            "zero_tail_start": 0,
            "freq_len": int(transform["freq_len"]),
            "quant_dtype": str(np.dtype(np.int8 if torch_dtype == torch.int8 else np.int16)),
            "quant_min": dtype_info.min,
            "quant_max": dtype_info.max,
        }
    rounded = torch.round(coefficients / table.view(_table_view(table, transform)))
    clip_mask = (rounded < dtype_info.min) | (rounded > dtype_info.max)
    clip_ratio = float(clip_mask.float().mean().item())
    clip_enabled = cfg.quant.clip_int8 if torch_dtype == torch.int8 else cfg.quant.clip_int16
    if clip_enabled:
        rounded = rounded.clamp(dtype_info.min, dtype_info.max)
    elif clip_ratio:
        raise ValueError(f"CacheJPEG {torch_dtype} overflow; increase q_global or enable clipping.")
    abs_values = rounded.abs()
    abs_max = float(abs_values.max().item())
    # Sampling bounds diagnostic overhead while keeping the quantization itself exact.
    sampled = abs_values.flatten()[:: max(1, abs_values.numel() // 200_000)]
    abs_p99 = float(torch.quantile(sampled.float(), 0.99).item())
    quantized = rounded.to(torch_dtype)
    zero_tail_start = _apply_zero_tail(quantized, cfg, transform)
    return quantized, {
        "nonzero_ratio": float((quantized != 0).float().mean().item()),
        "clip_ratio": clip_ratio,
        "abs_max_preclip": abs_max,
        "abs_p99_preclip": abs_p99,
        "zero_tail_start": int(zero_tail_start),
        "freq_len": int(transform["freq_len"]),
        "quant_dtype": str(np.dtype(np.int8 if torch_dtype == torch.int8 else np.int16)),
        "quant_min": dtype_info.min,
        "quant_max": dtype_info.max,
    }


def _pack_quantized(values: torch.Tensor, representation: str, backend: str) -> dict[str, Any]:
    array = values.detach().cpu().numpy()
    if representation == "adaptive_int":
        return pack_adaptive_int(array, backend=backend)
    if representation == "dense_int8":
        return pack_dense_int8(array, backend=backend)
    return pack_dense_int16(array, backend=backend)


def _unpack_quantized(packed: dict[str, Any], device: torch.device) -> torch.Tensor:
    if packed["representation"] == "adaptive_int":
        array = unpack_adaptive_int(packed)
    elif packed["representation"] == "dense_int8":
        array = unpack_dense_int8(packed)
    else:
        array = unpack_dense_int16(packed)
    return torch.from_numpy(array.copy()).to(device=device)


def _frequency_prune_settings(
    config: dict[str, Any],
    cfg: CacheJPEGConfig,
    transform: dict[str, Any],
) -> dict[str, Any] | None:
    prune_cfg = config.get("frequency_prune") or {}
    if not isinstance(prune_cfg, dict):
        raise ValueError("cachejpeg.frequency_prune must be a mapping.")
    if not bool(prune_cfg.get("enabled", False)):
        return None
    if cfg.block.mode != "global":
        raise ValueError("CacheJPEG frequency pruning currently requires block.mode=global.")

    prune_from = str(
        prune_cfg.get("prune_from", prune_cfg.get("prune_from_band", "B4"))
    ).strip().upper()
    freq_len = int(transform["freq_len"])
    keep_freq_len = int(frequency_band_slice(freq_len, prune_from).start)
    if keep_freq_len <= 0 or keep_freq_len >= freq_len:
        raise ValueError(
            f"frequency_prune.prune_from={prune_from!r} leaves an invalid "
            f"low-frequency prefix of {keep_freq_len} for freq_len={freq_len}."
        )
    return {
        "enabled": True,
        "prune_from": prune_from,
        "original_freq_len": freq_len,
        "stored_freq_len": keep_freq_len,
        "pruned_freq_len": freq_len - keep_freq_len,
        "retained_ratio": keep_freq_len / freq_len,
    }


def _truncate_quantized_frequencies(
    quantized: torch.Tensor,
    prune: dict[str, Any] | None,
) -> torch.Tensor:
    if prune is None:
        return quantized
    return quantized[:, :, : int(prune["stored_freq_len"]), :].contiguous()


def _restore_pruned_frequencies(
    quantized: torch.Tensor,
    transform: dict[str, Any],
) -> torch.Tensor:
    prune = transform.get("frequency_prune")
    if not prune or not bool(prune.get("enabled", False)):
        return quantized
    original_freq_len = int(prune["original_freq_len"])
    stored_freq_len = int(prune["stored_freq_len"])
    if quantized.ndim != 4:
        raise ValueError(
            "Pruned global CacheJPEG coefficients must have shape [B,H,F,D]."
        )
    if int(quantized.shape[2]) != stored_freq_len:
        raise ValueError(
            "Pruned CacheJPEG payload frequency length does not match its metadata: "
            f"{quantized.shape[2]} != {stored_freq_len}."
        )
    restored_shape = list(quantized.shape)
    restored_shape[2] = original_freq_len
    restored = torch.zeros(
        restored_shape,
        dtype=quantized.dtype,
        device=quantized.device,
    )
    restored[:, :, :stored_freq_len, :] = quantized
    return restored


class GPUCacheJPEGCodec:
    """CacheJPEG with FP32 DCT/quantization on GPU and entropy coding on CPU."""

    uses_gpu_transform = True

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)

    @staticmethod
    def _validate_supported_config(cfg: CacheJPEGConfig) -> None:
        if cfg.probe.enabled:
            raise ValueError("GPU CacheJPEG does not yet support probe/raw-except modes; use compute.backend=cpu.")

    def _encode_layer_data(
        self,
        layer_index: int,
        num_layers: int,
        key: torch.Tensor,
        value: torch.Tensor,
        cfg: CacheJPEGConfig,
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Encode one layer while retaining its global layer-dependent policy."""
        key = key.detach().to(device=self.device)
        value = value.detach().to(device=self.device)
        shape = tuple(int(item) for item in key.shape)
        seq_len = shape[2]
        anchors = _anchor_indices(seq_len, cfg, self.device)
        body = _body_indices(seq_len, anchors)
        layer_data: dict[str, Any] = {
            "layer_idx": layer_index,
            "shape": shape,
            "anchor_indices": anchors.cpu().numpy(),
            "body_indices": body.cpu().numpy(),
            "key_anchor_values": key.index_select(2, anchors).to(torch.float16).cpu().numpy(),
            "value_anchor_values": value.index_select(2, anchors).to(torch.float16).cpu().numpy(),
        }
        layer_diag: dict[str, Any] = {
            "layer_idx": layer_index,
            "layer_group": layer_group(layer_index, num_layers),
            "sink_count": min(seq_len, max(0, int(cfg.anchors.sink_count))),
            "recent_count": min(seq_len, max(0, int(cfg.anchors.recent_count))),
            "preserve_options": bool(cfg.anchors.preserve_options),
            "option_anchor_effective": bool(cfg.anchors.preserve_options and cfg.anchors.option_token_indices),
            "anchor_count": int(anchors.numel()),
            "anchor_ratio": float(anchors.numel() / max(seq_len, 1)),
            "body_count": int(body.numel()),
            "body_ratio": float(body.numel() / max(seq_len, 1)),
        }
        for component_name, tensor in (("key", key), ("value", value)):
            component_body = tensor.index_select(2, body)
            coefficients, transform = _forward_dct(component_body, cfg)
            table_numpy = build_frequency_table(
                int(transform["freq_len"]),
                cfg,
                layer_idx=layer_index,
                num_layers=num_layers,
                kv_type=component_name,
            )
            table = torch.from_numpy(table_numpy).to(device=self.device, dtype=torch.float32)
            quantized, quant_diag = _quantize(coefficients, table, cfg, transform)
            frequency_prune = _frequency_prune_settings(config, cfg, transform)
            stored_quantized = _truncate_quantized_frequencies(
                quantized, frequency_prune
            )
            if frequency_prune is not None:
                transform = {
                    **transform,
                    "frequency_prune": dict(frequency_prune),
                }
            layer_data[f"{component_name}_body"] = _pack_quantized(
                stored_quantized,
                cfg.entropy.representation,
                cfg.entropy.backend,
            )
            layer_data[f"{component_name}_transform"] = transform
            layer_data[f"{component_name}_quant_table"] = table_numpy
            layer_diag[component_name] = {
                **quant_diag,
                "quant_table_min": float(table_numpy.min()) if table_numpy.size else 0.0,
                "quant_table_max": float(table_numpy.max()) if table_numpy.size else 0.0,
                "quant_table_mean": float(table_numpy.mean()) if table_numpy.size else 0.0,
                "frequency_prune": (
                    dict(frequency_prune)
                    if frequency_prune is not None
                    else {"enabled": False}
                ),
            }
            del component_body, coefficients, table, quantized, stored_quantized
        layer_data["diagnostics"] = layer_diag
        return layer_data, layer_diag

    def encode_layer(
        self,
        layer_index: int,
        num_layers: int,
        key: torch.Tensor,
        value: torch.Tensor,
        config: dict[str, Any],
    ) -> EncodedPayload:
        """Encode a layer as an independently transportable payload."""
        cfg = resolve_cachejpeg_config(config)
        self._validate_supported_config(cfg)
        layer, diagnostics = self._encode_layer_data(
            layer_index, num_layers, key, value, cfg, config
        )
        component_bytes = _component_bytes([layer])
        summary = CacheJPEGCodec._summary(cfg, [layer], [diagnostics], component_bytes)
        summary.update({"compute_backend": "gpu", "transform_dtype": "float32"})
        prune_diag = diagnostics["key"]["frequency_prune"]
        summary["frequency_prune"] = dict(prune_diag)
        return EncodedPayload(
            payload={
                "method": "cachejpeg",
                "version": 2,
                "compute_backend": "gpu",
                "streaming": True,
                "layer_idx": int(layer_index),
                "num_layers": int(num_layers),
                "layers": [layer],
                "diagnostics": [diagnostics],
                "component_bytes": component_bytes,
                "summary": summary,
            },
            local_summary=summary,
        )

    def encode(self, past_key_values, config: dict[str, Any]) -> EncodedPayload:
        cfg = resolve_cachejpeg_config(config)
        self._validate_supported_config(cfg)
        layers: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        num_layers = len(past_key_values)
        for layer_index, (key, value) in enumerate(past_key_values):
            layer_data, layer_diag = self._encode_layer_data(
                layer_index, num_layers, key, value, cfg, config
            )
            layers.append(layer_data)
            diagnostics.append(layer_diag)

        component_bytes = _component_bytes(layers)
        summary = CacheJPEGCodec._summary(cfg, layers, diagnostics, component_bytes)
        summary["compute_backend"] = "gpu"
        summary["transform_dtype"] = "float32"
        if diagnostics:
            summary["frequency_prune"] = dict(
                diagnostics[0]["key"]["frequency_prune"]
            )
        if cfg.entropy.representation == "adaptive_int":
            width_counts = {4: 0, 8: 0, 16: 0}
            source_int16_bytes = 0
            adaptive_raw_bytes = 0
            for layer in layers:
                for component_name in ("key_body", "value_body"):
                    packed = layer[component_name]
                    for width, count in packed["bit_width_counts"].items():
                        width_counts[int(width)] += int(count)
                    source_int16_bytes += int(packed["source_int16_bytes"])
                    adaptive_raw_bytes += int(packed["raw_bytes"])
            summary["adaptive_granularity"] = "frequency_band"
            summary["adaptive_bit_width_counts"] = width_counts
            summary["adaptive_source_int16_bytes"] = source_int16_bytes
            summary["adaptive_packed_raw_bytes"] = adaptive_raw_bytes
            summary["adaptive_raw_reduction_ratio"] = (
                1.0 - adaptive_raw_bytes / source_int16_bytes if source_int16_bytes else 0.0
            )
        return EncodedPayload(
            payload={
                "method": "cachejpeg",
                "version": 2,
                "compute_backend": "gpu",
                "layers": layers,
                "diagnostics": diagnostics,
                "component_bytes": component_bytes,
                "summary": summary,
            },
            local_summary=summary,
        )

    def decode(self, payload: EncodedPayload, config: dict[str, Any]):
        restored = []
        for layer in payload.payload["layers"]:
            shape = tuple(int(item) for item in layer["shape"])
            anchors = torch.from_numpy(np.asarray(layer["anchor_indices"], dtype=np.int64)).to(self.device)
            body = torch.from_numpy(np.asarray(layer["body_indices"], dtype=np.int64)).to(self.device)
            components = []
            for component_name in ("key", "value"):
                quantized = _unpack_quantized(layer[f"{component_name}_body"], self.device)
                transform = layer[f"{component_name}_transform"]
                quantized = _restore_pruned_frequencies(quantized, transform)
                table = torch.from_numpy(
                    np.asarray(layer[f"{component_name}_quant_table"], dtype=np.float32)
                ).to(self.device)
                coefficients = quantized.float() * table.view(_table_view(table, transform))
                body_values = _inverse_dct(coefficients, transform)
                values = torch.zeros(shape, device=self.device, dtype=torch.float32)
                if body.numel():
                    values.index_copy_(2, body, body_values)
                if anchors.numel():
                    anchor_values = torch.from_numpy(
                        np.asarray(layer[f"{component_name}_anchor_values"], dtype=np.float16)
                    ).to(device=self.device, dtype=torch.float32)
                    values.index_copy_(2, anchors, anchor_values)
                components.append(values)
            restored.append(tuple(components))
        return tuple(restored)
