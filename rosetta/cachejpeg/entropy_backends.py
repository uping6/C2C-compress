from __future__ import annotations

import zlib
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import lz4.frame as lz4_frame
except ImportError:  # pragma: no cover
    lz4_frame = None


SUPPORTED_ENTROPY_BACKENDS = {"none", "lz4", "zigzag_rle", "zigzag_rle_lz4"}


def validate_entropy_backend(backend: str) -> str:
    normalized = str(backend).lower()
    if normalized in SUPPORTED_ENTROPY_BACKENDS:
        return normalized
    if normalized.startswith("zlib"):
        suffix = normalized.removeprefix("zlib")
        if suffix == "":
            return normalized
        try:
            level = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Invalid CacheJPEG zlib backend: {backend}") from exc
        if -1 <= level <= 9:
            return normalized
    supported = "none, zlib[0-9], lz4, zigzag_rle, zigzag_rle_lz4"
    raise ValueError(f"Unsupported CacheJPEG entropy backend {backend!r}; expected one of: {supported}.")


def _zlib_level(backend: str) -> int | None:
    if not backend.startswith("zlib"):
        return None
    suffix = backend.removeprefix("zlib")
    return 6 if suffix == "" else int(suffix)


def _require_lz4():
    if lz4_frame is None:
        raise ImportError("CacheJPEG LZ4 backend requires `pip install lz4`.")
    return lz4_frame


@lru_cache(maxsize=64)
def zigzag_linear_indices(rows: int, columns: int) -> np.ndarray:
    """Classic diagonal zigzag order for the final two tensor dimensions."""
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Zigzag dimensions must be positive, got {(rows, columns)}")
    indices: list[int] = []
    for diagonal in range(rows + columns - 1):
        row_min = max(0, diagonal - columns + 1)
        row_max = min(rows - 1, diagonal)
        row_iter = range(row_max, row_min - 1, -1) if diagonal % 2 == 0 else range(row_min, row_max + 1)
        for row in row_iter:
            indices.append(row * columns + diagonal - row)
    result = np.asarray(indices, dtype=np.int64)
    result.setflags(write=False)
    return result


def _zigzag_flatten(values: np.ndarray) -> np.ndarray:
    rows, columns = int(values.shape[-2]), int(values.shape[-1])
    order = zigzag_linear_indices(rows, columns)
    return values.reshape(-1, rows * columns)[:, order].reshape(-1)


def _zigzag_restore(sequence: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    rows, columns = int(shape[-2]), int(shape[-1])
    order = zigzag_linear_indices(rows, columns)
    encoded = sequence.reshape(-1, rows * columns)
    restored = np.empty_like(encoded)
    restored[:, order] = encoded
    return restored.reshape(shape)


def _zero_rle_encode(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nonzero_indices = np.flatnonzero(sequence)
    if nonzero_indices.size:
        runs_before = np.diff(np.concatenate((np.asarray([-1], dtype=np.int64), nonzero_indices))) - 1
        trailing = int(sequence.size - int(nonzero_indices[-1]) - 1)
        runs = np.concatenate((runs_before, np.asarray([trailing], dtype=np.int64)))
        nonzero_values = sequence[nonzero_indices]
    else:
        runs = np.asarray([sequence.size], dtype=np.int64)
        nonzero_values = np.empty(0, dtype=sequence.dtype)
    if int(runs.max(initial=0)) > np.iinfo(np.uint32).max:
        raise ValueError("CacheJPEG RLE zero run exceeds uint32 capacity.")
    return runs.astype("<u4", copy=False), nonzero_values


def _zero_rle_decode(runs: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    if runs.size != values.size + 1:
        raise ValueError("Invalid CacheJPEG RLE stream: run/value count mismatch.")
    result = np.zeros(int(size), dtype=values.dtype)
    if values.size:
        positions = np.cumsum(runs[:-1], dtype=np.int64) + np.arange(values.size, dtype=np.int64)
        if int(positions[-1]) >= int(size):
            raise ValueError("Invalid CacheJPEG RLE stream: decoded position exceeds tensor size.")
        result[positions] = values
    decoded_size = int(runs.astype(np.uint64).sum()) + int(values.size)
    if decoded_size != int(size):
        raise ValueError(f"Invalid CacheJPEG RLE stream size: {decoded_size} != {size}.")
    return result


def _pack(values: np.ndarray, *, backend: str, representation: str) -> dict[str, Any]:
    backend = validate_entropy_backend(backend)
    raw = values.tobytes(order="C")
    metadata: dict[str, Any] = {}
    zlib_level = _zlib_level(backend)
    if backend == "none":
        data: bytes | np.ndarray = values.copy()
        stored_bytes = int(values.nbytes)
    elif zlib_level is not None:
        data = zlib.compress(raw, level=zlib_level)
        stored_bytes = len(data)
    elif backend == "lz4":
        data = _require_lz4().compress(raw, compression_level=0)
        stored_bytes = len(data)
    else:
        zigzagged = _zigzag_flatten(values)
        runs, nonzero_values = _zero_rle_encode(zigzagged)
        rle_bytes = runs.tobytes(order="C") + nonzero_values.tobytes(order="C")
        data = _require_lz4().compress(rle_bytes, compression_level=0) if backend == "zigzag_rle_lz4" else rle_bytes
        stored_bytes = len(data)
        metadata = {
            "scan_order": "zigzag_last_two_dims",
            "rle_symbol": 0,
            "rle_run_dtype": "uint32",
            "rle_run_count": int(runs.size),
            "rle_nonzero_count": int(nonzero_values.size),
        }
    return {
        "representation": representation,
        "backend": backend,
        "shape": tuple(int(x) for x in values.shape),
        "dtype": str(values.dtype),
        "data": data,
        "stored_bytes": int(stored_bytes),
        "raw_bytes": int(values.nbytes),
        **metadata,
    }


def _unpack(packed: dict[str, Any], dtype: np.dtype) -> np.ndarray:
    backend = validate_entropy_backend(str(packed["backend"]))
    shape = tuple(int(x) for x in packed["shape"])
    if backend == "none":
        return np.asarray(packed["data"], dtype=dtype).reshape(shape)
    if _zlib_level(backend) is not None:
        return np.frombuffer(zlib.decompress(packed["data"]), dtype=dtype).reshape(shape)
    if backend == "lz4":
        return np.frombuffer(_require_lz4().decompress(packed["data"]), dtype=dtype).reshape(shape)

    raw = _require_lz4().decompress(packed["data"]) if backend == "zigzag_rle_lz4" else packed["data"]
    run_count = int(packed["rle_run_count"])
    nonzero_count = int(packed["rle_nonzero_count"])
    run_bytes = run_count * np.dtype("<u4").itemsize
    expected_bytes = run_bytes + nonzero_count * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(f"Invalid CacheJPEG RLE payload byte length: {len(raw)} != {expected_bytes}.")
    runs = np.frombuffer(raw, dtype="<u4", count=run_count)
    values = np.frombuffer(raw, dtype=dtype, count=nonzero_count, offset=run_bytes)
    sequence = _zero_rle_decode(runs, values, int(np.prod(shape)))
    return _zigzag_restore(sequence, shape)


def pack_dense_int16(values: np.ndarray, *, backend: str) -> dict[str, Any]:
    return _pack(values.astype(np.int16, copy=False), backend=backend, representation="dense_int16")


def pack_dense_int8(values: np.ndarray, *, backend: str) -> dict[str, Any]:
    return _pack(values.astype(np.int8, copy=False), backend=backend, representation="dense_int8")


def unpack_dense_int16(packed: dict[str, Any]) -> np.ndarray:
    return _unpack(packed, np.dtype(np.int16))


def unpack_dense_int8(packed: dict[str, Any]) -> np.ndarray:
    return _unpack(packed, np.dtype(np.int8))


def _frequency_band_slices(freq_len: int) -> list[tuple[str, int, int]]:
    raw_edges = [
        0,
        1,
        max(2, freq_len // 64),
        max(4, freq_len // 16),
        max(8, freq_len // 4),
        max(16, freq_len // 2),
        freq_len,
    ]
    edges: list[int] = []
    previous = 0
    for raw_edge in raw_edges:
        edge = min(freq_len, max(previous, int(raw_edge)))
        edges.append(edge)
        previous = edge
    return [
        (f"B{index}", edges[index], edges[index + 1])
        for index in range(6)
        if edges[index + 1] > edges[index]
    ]


def _select_bit_width(values: np.ndarray) -> int:
    if values.size == 0:
        return 4
    minimum = int(values.min())
    maximum = int(values.max())
    if minimum >= -8 and maximum <= 7:
        return 4
    if minimum >= -128 and maximum <= 127:
        return 8
    if minimum >= -32768 and maximum <= 32767:
        return 16
    raise ValueError(f"Adaptive integer value range [{minimum}, {maximum}] exceeds INT16.")


def _pack_integer_values(values: np.ndarray, bit_width: int) -> bytes:
    flattened = values.reshape(-1)
    if bit_width == 16:
        return flattened.astype("<i2", copy=False).tobytes(order="C")
    if bit_width == 8:
        return flattened.astype(np.int8, copy=False).tobytes(order="C")
    if bit_width != 4:
        raise ValueError(f"Unsupported adaptive integer bit width: {bit_width}")
    nibbles = (flattened.astype(np.int16, copy=False) & 0x0F).astype(np.uint8)
    if nibbles.size % 2:
        nibbles = np.concatenate((nibbles, np.zeros(1, dtype=np.uint8)))
    return (nibbles[0::2] | (nibbles[1::2] << 4)).tobytes(order="C")


def _unpack_integer_values(data: bytes, bit_width: int, count: int) -> np.ndarray:
    if bit_width == 16:
        return np.frombuffer(data, dtype="<i2", count=count).astype(np.int16, copy=False)
    if bit_width == 8:
        return np.frombuffer(data, dtype=np.int8, count=count)
    if bit_width != 4:
        raise ValueError(f"Unsupported adaptive integer bit width: {bit_width}")
    packed = np.frombuffer(data, dtype=np.uint8)
    nibbles = np.empty(packed.size * 2, dtype=np.uint8)
    nibbles[0::2] = packed & 0x0F
    nibbles[1::2] = packed >> 4
    signed = nibbles.astype(np.int8)
    signed[signed >= 8] -= 16
    return signed[:count]


def _compress_bytes(raw: bytes, backend: str) -> bytes:
    zlib_level = _zlib_level(backend)
    if backend == "none":
        return raw
    if zlib_level is not None:
        return zlib.compress(raw, level=zlib_level)
    if backend == "lz4":
        return _require_lz4().compress(raw, compression_level=0)
    raise ValueError(f"Backend {backend!r} requires the adaptive RLE path.")


def _decompress_bytes(data: bytes, backend: str) -> bytes:
    if backend == "none":
        return data
    if _zlib_level(backend) is not None:
        return zlib.decompress(data)
    if backend == "lz4":
        return _require_lz4().decompress(data)
    raise ValueError(f"Backend {backend!r} requires the adaptive RLE path.")


def pack_adaptive_int(values: np.ndarray, *, backend: str) -> dict[str, Any]:
    """Losslessly store INT16 quantized coefficients with per-band 4/8/16-bit widths."""
    backend = validate_entropy_backend(backend)
    values = values.astype(np.int16, copy=False)
    frequency_axis = 2 if values.ndim == 4 else 3 if values.ndim == 5 else None
    if frequency_axis is None:
        raise ValueError(f"Adaptive CacheJPEG expects 4-D or 5-D coefficients, got {values.shape}.")
    bands = []
    raw_bytes = 0
    stored_bytes = 0
    bit_width_counts = {4: 0, 8: 0, 16: 0}
    for name, start, stop in _frequency_band_slices(int(values.shape[frequency_axis])):
        index = [slice(None)] * values.ndim
        index[frequency_axis] = slice(start, stop)
        band_values = np.ascontiguousarray(values[tuple(index)])
        bit_width = _select_bit_width(band_values)
        bit_width_counts[bit_width] += 1
        metadata: dict[str, Any] = {}
        if backend in {"zigzag_rle", "zigzag_rle_lz4"}:
            zigzagged = _zigzag_flatten(band_values)
            runs, nonzero_values = _zero_rle_encode(zigzagged)
            encoded_values = _pack_integer_values(nonzero_values, bit_width)
            raw = runs.tobytes(order="C") + encoded_values
            data = _require_lz4().compress(raw, compression_level=0) if backend == "zigzag_rle_lz4" else raw
            metadata = {
                "scan_order": "zigzag_last_two_dims",
                "rle_run_count": int(runs.size),
                "rle_nonzero_count": int(nonzero_values.size),
                "rle_run_bytes": int(runs.nbytes),
            }
        else:
            raw = _pack_integer_values(band_values, bit_width)
            data = _compress_bytes(raw, backend)
        raw_bytes += len(raw)
        stored_bytes += len(data)
        bands.append(
            {
                "name": name,
                "start": start,
                "stop": stop,
                "shape": tuple(int(item) for item in band_values.shape),
                "value_count": int(band_values.size),
                "bit_width": bit_width,
                "data": data,
                "raw_bytes": len(raw),
                "stored_bytes": len(data),
                **metadata,
            }
        )
    return {
        "representation": "adaptive_int",
        "backend": backend,
        "shape": tuple(int(item) for item in values.shape),
        "dtype": "adaptive_int4_int8_int16",
        "frequency_axis": frequency_axis,
        "granularity": "frequency_band",
        "candidates": (4, 8, 16),
        "bands": bands,
        "bit_width_counts": bit_width_counts,
        "data": b"",
        "stored_bytes": int(stored_bytes),
        "raw_bytes": int(raw_bytes),
        "source_int16_bytes": int(values.nbytes),
    }


def unpack_adaptive_int(packed: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(item) for item in packed["shape"])
    frequency_axis = int(packed["frequency_axis"])
    backend = validate_entropy_backend(str(packed["backend"]))
    restored = np.zeros(shape, dtype=np.int16)
    for band in packed["bands"]:
        bit_width = int(band["bit_width"])
        if backend in {"zigzag_rle", "zigzag_rle_lz4"}:
            raw = _require_lz4().decompress(band["data"]) if backend == "zigzag_rle_lz4" else band["data"]
            run_bytes = int(band["rle_run_bytes"])
            runs = np.frombuffer(raw[:run_bytes], dtype="<u4", count=int(band["rle_run_count"]))
            nonzero_values = _unpack_integer_values(
                raw[run_bytes:], bit_width, int(band["rle_nonzero_count"])
            ).astype(np.int16)
            sequence = _zero_rle_decode(runs, nonzero_values, int(band["value_count"]))
            band_values = _zigzag_restore(sequence, tuple(int(item) for item in band["shape"]))
        else:
            raw = _decompress_bytes(band["data"], backend)
            band_values = _unpack_integer_values(raw, bit_width, int(band["value_count"])).astype(np.int16)
            band_values = band_values.reshape(tuple(int(item) for item in band["shape"]))
        index = [slice(None)] * len(shape)
        index[frequency_axis] = slice(int(band["start"]), int(band["stop"]))
        restored[tuple(index)] = band_values
    return restored


def _extended_quant_dtype(representation: str) -> np.dtype:
    if representation == "dense_int8":
        return np.dtype(np.int8)
    if representation in {"dense_int16", "adaptive_int"}:
        return np.dtype(np.int16)
    raise ValueError(f"Unsupported CacheJPEG entropy representation: {representation}")


def _extended_pack_quantized(values: np.ndarray, *, representation: str, backend: str) -> dict[str, Any]:
    if representation == "adaptive_int":
        return pack_adaptive_int(values, backend=backend)
    if representation == "dense_int8":
        return pack_dense_int8(values, backend=backend)
    if representation == "dense_int16":
        return pack_dense_int16(values, backend=backend)
    raise ValueError(f"Unsupported CacheJPEG entropy representation: {representation}")


def _extended_unpack_quantized(packed: dict[str, Any]) -> np.ndarray:
    representation = str(packed.get("representation", "dense_int16"))
    if representation == "adaptive_int":
        return unpack_adaptive_int(packed)
    if representation == "dense_int8":
        return unpack_dense_int8(packed)
    if representation == "dense_int16":
        return unpack_dense_int16(packed)
    raise ValueError(f"Unsupported CacheJPEG entropy representation: {representation}")


def install_homo_cachejpeg_entropy_backends() -> None:
    """Install project-local entropy adapters into HomoC2C's codec module."""
    from homo_c2c_kv.codec.cachejpeg import codec as codec_module

    codec_module.pack_dense_int16 = pack_dense_int16
    codec_module.pack_dense_int8 = pack_dense_int8
    codec_module.unpack_dense_int16 = unpack_dense_int16
    codec_module.unpack_dense_int8 = unpack_dense_int8
    codec_module._quant_dtype = _extended_quant_dtype
    codec_module._pack_quantized = _extended_pack_quantized
    codec_module._unpack_quantized = _extended_unpack_quantized
