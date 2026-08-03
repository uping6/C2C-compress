import numpy as np
import torch
from scipy.fft import dct, idct

from rosetta.cachejpeg.wrapper import _ensure_homo_imports

_ensure_homo_imports("/data/smy/HomoC2C-KV/src")

from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec, dct_ii_ortho, idct_iii_ortho


def test_torch_dct_matches_scipy_for_odd_and_even_lengths():
    generator = torch.Generator().manual_seed(17)
    for length in (7, 8, 31, 64):
        values = torch.randn(2, 3, length, 5, generator=generator, dtype=torch.float32)
        actual = dct_ii_ortho(values, axis=2)
        expected = dct(values.numpy(), axis=2, norm="ortho")
        assert np.allclose(actual.numpy(), expected, atol=2e-5, rtol=2e-5)


def test_torch_idct_matches_scipy_and_reconstructs_input():
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(2, 3, 33, 4, generator=generator, dtype=torch.float32)
    coefficients = dct_ii_ortho(values, axis=2)
    restored = idct_iii_ortho(coefficients, axis=2)
    scipy_restored = idct(coefficients.numpy(), axis=2, norm="ortho")
    assert np.allclose(restored.numpy(), scipy_restored, atol=2e-5, rtol=2e-5)
    assert torch.allclose(restored, values, atol=2e-5, rtol=2e-5)


def test_gpu_codec_matches_cpu_codec_reconstruction_on_cuda():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    codec_cls, _, _, _ = _ensure_homo_imports("/data/smy/HomoC2C-KV/src")
    cpu_codec = codec_cls()
    gpu_codec = GPUCacheJPEGCodec(device=device)
    generator = torch.Generator().manual_seed(29)
    cpu_cache = tuple(
        (
            torch.randn(1, 2, 65, 8, generator=generator),
            torch.randn(1, 2, 65, 8, generator=generator),
        )
        for _ in range(2)
    )
    gpu_cache = tuple((key.to(device=device, dtype=torch.bfloat16), value.to(device=device, dtype=torch.bfloat16)) for key, value in cpu_cache)
    config = {
        "method": "cachejpeg",
        "anchors": {"sink_count": 1},
        "block": {"mode": "global"},
        "quant": {"q_global": 1.0, "low": 1.0, "high": 8.0, "curve": "quadratic"},
        "entropy": {"representation": "dense_int16", "backend": "lz4"},
    }
    # Use the same BF16-derived values on both paths so only implementation differs.
    cpu_bf16_cache = tuple((key.float().cpu(), value.float().cpu()) for key, value in gpu_cache)
    cpu_payload = cpu_codec.encode(cpu_bf16_cache, config)
    gpu_payload = gpu_codec.encode(gpu_cache, config)
    cpu_restored = cpu_codec.decode(cpu_payload, config)
    gpu_restored = gpu_codec.decode(gpu_payload, config)

    assert gpu_payload.local_summary["compute_backend"] == "gpu"
    assert gpu_payload.local_summary["transform_dtype"] == "float32"
    for cpu_layer, gpu_layer in zip(cpu_restored, gpu_restored):
        assert torch.allclose(gpu_layer[0].cpu(), cpu_layer[0], atol=2e-4, rtol=2e-4)
        assert torch.allclose(gpu_layer[1].cpu(), cpu_layer[1], atol=2e-4, rtol=2e-4)


def test_gpu_codec_adaptive_int_matches_dense_int16_exactly():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    codec = GPUCacheJPEGCodec(device=device)
    generator = torch.Generator().manual_seed(43)
    cache = ((
        torch.randn(1, 2, 97, 8, generator=generator).to(device=device, dtype=torch.bfloat16),
        torch.randn(1, 2, 97, 8, generator=generator).to(device=device, dtype=torch.bfloat16),
    ),)
    base_config = {
        "method": "cachejpeg",
        "anchors": {"sink_count": 1},
        "block": {"mode": "global"},
        "entropy": {"backend": "lz4"},
    }
    dense_config = {**base_config, "entropy": {"backend": "lz4", "representation": "dense_int16"}}
    adaptive_config = {**base_config, "entropy": {"backend": "lz4", "representation": "adaptive_int"}}
    dense = codec.decode(codec.encode(cache, dense_config), dense_config)
    adaptive_payload = codec.encode(cache, adaptive_config)
    adaptive = codec.decode(adaptive_payload, adaptive_config)
    assert torch.equal(dense[0][0], adaptive[0][0])
    assert torch.equal(dense[0][1], adaptive[0][1])
    assert adaptive_payload.local_summary["adaptive_granularity"] == "frequency_band"
    assert sum(adaptive_payload.local_summary["adaptive_bit_width_counts"].values()) == 12


def test_encode_layer_preserves_global_layer_policy_and_decodes_one_layer():
    codec = GPUCacheJPEGCodec(device="cpu")
    generator = torch.Generator().manual_seed(47)
    key = torch.randn(1, 2, 33, 4, generator=generator)
    value = torch.randn(1, 2, 33, 4, generator=generator)
    config = {
        "anchors": {"sink_count": 1},
        "block": {"mode": "global"},
        "quant": {"layer_group_scales": {"early": 1.0, "middle": 2.0, "late": 3.0}},
        "entropy": {"representation": "dense_int16", "backend": "zlib1"},
    }

    payload = codec.encode_layer(7, 9, key, value, config)
    restored = codec.decode(payload, config)

    assert payload.payload["layer_idx"] == 7
    assert payload.payload["num_layers"] == 9
    assert payload.payload["layers"][0]["layer_idx"] == 7
    assert payload.payload["diagnostics"][0]["layer_group"] == "late"
    assert len(restored) == 1
    assert restored[0][0].shape == key.shape


def test_global_frequency_prune_stores_only_low_frequency_prefix_and_zero_fills_decode():
    codec = GPUCacheJPEGCodec(device="cpu")
    generator = torch.Generator().manual_seed(53)
    key = torch.randn(1, 2, 65, 4, generator=generator)
    value = torch.randn(1, 2, 65, 4, generator=generator)
    base_config = {
        "anchors": {"sink_count": 1},
        "block": {"mode": "global"},
        "quant": {"low": 1.0, "high": 1.0},
        "entropy": {"representation": "dense_int16", "backend": "none"},
    }
    prune_config = {
        **base_config,
        "frequency_prune": {"enabled": True, "prune_from": "B4"},
    }

    full_payload = codec.encode(((key, value),), base_config)
    pruned_payload = codec.encode(((key, value),), prune_config)
    restored = codec.decode(pruned_payload, prune_config)

    layer = pruned_payload.payload["layers"][0]
    transform = layer["key_transform"]
    prune = transform["frequency_prune"]
    assert prune["original_freq_len"] == 64
    assert prune["stored_freq_len"] == 16
    assert prune["pruned_freq_len"] == 48
    assert layer["key_body"]["shape"] == (1, 2, 16, 4)
    assert layer["value_body"]["shape"] == (1, 2, 16, 4)
    assert (
        pruned_payload.payload["component_bytes"]["body_stored_bytes"]
        < full_payload.payload["component_bytes"]["body_stored_bytes"]
    )
    assert restored[0][0].shape == key.shape
    assert restored[0][1].shape == value.shape

    key_body = key[:, :, 1:, :]
    quantized = torch.round(dct_ii_ortho(key_body, axis=2)).to(torch.int16)
    quantized[:, :, 16:, :] = 0
    expected_body = idct_iii_ortho(quantized.float(), axis=2)
    assert torch.equal(restored[0][0][:, :, :1, :], key[:, :, :1, :].half().float())
    assert torch.allclose(restored[0][0][:, :, 1:, :], expected_body, atol=2e-5, rtol=2e-5)


def test_frequency_prune_rejects_fixed_block_mode():
    codec = GPUCacheJPEGCodec(device="cpu")
    cache = ((torch.zeros(1, 1, 65, 2), torch.zeros(1, 1, 65, 2)),)
    config = {
        "block": {"mode": "fixed", "size": 64},
        "frequency_prune": {"enabled": True, "prune_from": "B4"},
        "entropy": {"representation": "dense_int16", "backend": "none"},
    }

    with np.testing.assert_raises_regex(ValueError, "block.mode=global"):
        codec.encode(cache, config)
