import numpy as np
import torch

from rosetta.cachejpeg.config import resolve_cachejpeg_eval_config
from rosetta.cachejpeg.entropy_backends import (
    pack_adaptive_int,
    pack_dense_int16,
    pack_dense_int8,
    unpack_dense_int16,
    unpack_dense_int8,
    unpack_adaptive_int,
    zigzag_linear_indices,
)
from rosetta.cachejpeg.wrapper import _ensure_homo_imports


BACKENDS = ("zlib1", "lz4", "zigzag_rle", "zigzag_rle_lz4")


def test_zigzag_order_for_three_by_three_matrix():
    assert zigzag_linear_indices(3, 3).tolist() == [0, 1, 3, 6, 4, 2, 5, 7, 8]


def test_entropy_backends_roundtrip_dense_int16_and_int8():
    rng = np.random.default_rng(7)
    values16 = rng.integers(-20, 21, size=(2, 3, 9, 5), dtype=np.int16)
    values16[rng.random(values16.shape) < 0.72] = 0
    values8 = values16.astype(np.int8)

    for backend in BACKENDS:
        packed16 = pack_dense_int16(values16, backend=backend)
        packed8 = pack_dense_int8(values8, backend=backend)
        assert np.array_equal(unpack_dense_int16(packed16), values16)
        assert np.array_equal(unpack_dense_int8(packed8), values8)
        assert packed16["backend"] == backend
        assert packed16["stored_bytes"] == (
            packed16["data"].nbytes if isinstance(packed16["data"], np.ndarray) else len(packed16["data"])
        )


def test_zigzag_rle_supports_fixed_block_coefficient_shape():
    values = np.zeros((1, 2, 4, 8, 6), dtype=np.int16)
    values[..., 0, 0] = 3
    values[..., 2, 1] = -4
    for backend in ("zigzag_rle", "zigzag_rle_lz4"):
        packed = pack_dense_int16(values, backend=backend)
        assert packed["scan_order"] == "zigzag_last_two_dims"
        assert np.array_equal(unpack_dense_int16(packed), values)


def test_cachejpeg_codec_roundtrip_is_identical_across_entropy_backends():
    codec_cls, _, _, _ = _ensure_homo_imports("/data/smy/HomoC2C-KV/src")
    codec = codec_cls()
    generator = torch.Generator().manual_seed(11)
    cache = tuple(
        (
            torch.randn(1, 2, 17, 8, generator=generator),
            torch.randn(1, 2, 17, 8, generator=generator),
        )
        for _ in range(2)
    )
    reconstructed = {}
    for backend in BACKENDS:
        config = {
            "method": "cachejpeg",
            "anchors": {"sink_count": 1},
            "block": {"mode": "global"},
            "entropy": {"representation": "dense_int16", "backend": backend},
        }
        payload = codec.encode(cache, config)
        reconstructed[backend] = codec.decode(payload, config)
        assert payload.local_summary["entropy_backend"] == backend

    reference = reconstructed["zlib1"]
    for backend in BACKENDS[1:]:
        for reference_layer, candidate_layer in zip(reference, reconstructed[backend]):
            assert torch.equal(reference_layer[0], candidate_layer[0])
            assert torch.equal(reference_layer[1], candidate_layer[1])


def test_cachejpeg_config_accepts_new_backends_and_rejects_unknown_backend():
    for backend in BACKENDS:
        config = resolve_cachejpeg_eval_config({"entropy": {"backend": backend}})
        assert config.entropy.backend == backend

    try:
        resolve_cachejpeg_eval_config({"entropy": {"backend": "zip-fast"}})
    except ValueError as exc:
        assert "Unsupported CacheJPEG entropy backend" in str(exc)
    else:
        raise AssertionError("Unknown entropy backend should fail fast.")


def test_adaptive_int_selects_4_8_16_bits_by_frequency_band_without_value_changes():
    values = np.zeros((1, 2, 64, 7), dtype=np.int16)
    values[:, :, 0, 0] = 300
    values[:, :, 1, 1] = -100
    values[:, :, 2:, :] = np.arange(-7, 8, dtype=np.int16)[
        np.arange(values[:, :, 2:, :].size).reshape(values[:, :, 2:, :].shape) % 15
    ]
    packed = pack_adaptive_int(values, backend="none")
    widths = {band["name"]: band["bit_width"] for band in packed["bands"]}
    assert widths["B0"] == 16
    assert widths["B1"] == 8
    assert 4 in widths.values()
    assert packed["raw_bytes"] < packed["source_int16_bytes"]
    assert np.array_equal(unpack_adaptive_int(packed), values)


def test_adaptive_int_roundtrips_with_every_entropy_backend_and_fixed_block_shape():
    rng = np.random.default_rng(37)
    values = rng.integers(-7, 8, size=(1, 2, 3, 16, 5), dtype=np.int16)
    values[rng.random(values.shape) < 0.75] = 0
    values[..., 0, 0] = 1000
    for backend in ("none",) + BACKENDS:
        packed = pack_adaptive_int(values, backend=backend)
        assert packed["representation"] == "adaptive_int"
        assert packed["granularity"] == "frequency_band"
        assert np.array_equal(unpack_adaptive_int(packed), values)


def test_complete_codec_adaptive_int_matches_dense_int16_reconstruction():
    codec_cls, _, _, _ = _ensure_homo_imports("/data/smy/HomoC2C-KV/src")
    codec = codec_cls()
    generator = torch.Generator().manual_seed(41)
    cache = ((torch.randn(1, 2, 33, 8, generator=generator), torch.randn(1, 2, 33, 8, generator=generator)),)
    base = {
        "method": "cachejpeg",
        "anchors": {"sink_count": 1},
        "block": {"mode": "global"},
        "entropy": {"backend": "lz4"},
    }
    dense_config = {**base, "entropy": {"backend": "lz4", "representation": "dense_int16"}}
    adaptive_config = {**base, "entropy": {"backend": "lz4", "representation": "adaptive_int"}}
    dense = codec.decode(codec.encode(cache, dense_config), dense_config)
    adaptive_payload = codec.encode(cache, adaptive_config)
    adaptive = codec.decode(adaptive_payload, adaptive_config)
    assert adaptive_payload.local_summary["entropy_representation"] == "adaptive_int"
    assert torch.equal(dense[0][0], adaptive[0][0])
    assert torch.equal(dense[0][1], adaptive[0][1])
