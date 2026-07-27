import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.latent_kv import (
    LatentKVBridge,
    LatentKVCompressor,
    build_proportional_layer_mapping,
)
from rosetta.model.projector import load_projector, save_projector


def _inputs(dtype=torch.float32, sequence_length=16):
    return (
        torch.randn(2, 4, sequence_length, 32, dtype=dtype),
        torch.randn(2, 4, sequence_length, 32, dtype=dtype),
        torch.randn(2, 2, sequence_length, 64, dtype=dtype),
        torch.randn(2, 2, sequence_length, 64, dtype=dtype),
    )


def _compressor(dtype=torch.float32, residual_scale=0.0):
    return LatentKVCompressor(
        sharer_num_kv_heads=4,
        sharer_head_dim=32,
        receiver_num_kv_heads=2,
        receiver_head_dim=64,
        latent_dim=128,
        mlp_ratio=2.0,
        init_residual_scale=residual_scale,
        dtype=dtype,
    )


def _dynamic_cache(layers):
    cache = DynamicCache()
    for key, value in layers:
        cache.key_cache.append(key)
        cache.value_cache.append(value)
    return cache


def test_latent_kv_compressor_shape_and_identity_initialization():
    sharer_key, sharer_value, receiver_key, receiver_value = _inputs()
    module = _compressor()

    fused_key, fused_value = module(
        (sharer_key, sharer_value), (receiver_key, receiver_value)
    )

    assert fused_key.shape == receiver_key.shape
    assert fused_value.shape == receiver_value.shape
    assert torch.allclose(fused_key, receiver_key, atol=1e-7, rtol=0.0)
    assert torch.allclose(fused_value, receiver_value, atol=1e-7, rtol=0.0)
    assert module.joint_input_dim == 512
    assert module.last_stats["residual_receiver_ratio"] == 0.0


def test_latent_kv_compressor_gradients_and_frozen_models():
    module = _compressor(residual_scale=0.1)
    frozen_sharer = torch.nn.Linear(2, 2).requires_grad_(False)
    frozen_receiver = torch.nn.Linear(2, 2).requires_grad_(False)
    inputs = _inputs(sequence_length=8)

    outputs = module((inputs[0], inputs[1]), (inputs[2], inputs[3]))
    sum(tensor.square().mean() for tensor in outputs).backward()

    assert module.down_proj.weight.grad is not None
    assert module.key_up_proj.weight.grad is not None
    assert module.value_up_proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in frozen_sharer.parameters())
    assert all(parameter.grad is None for parameter in frozen_receiver.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_latent_kv_compressor_dtype_and_numerical_stability(dtype):
    module = _compressor(dtype=dtype, residual_scale=0.1)
    inputs = _inputs(dtype=dtype, sequence_length=8)

    outputs = module((inputs[0], inputs[1]), (inputs[2], inputs[3]))

    assert all(tensor.dtype == dtype for tensor in outputs)
    assert all(torch.isfinite(tensor).all() for tensor in outputs)


@pytest.mark.parametrize("sequence_length", [1, 8, 128])
def test_latent_kv_compressor_sequence_lengths(sequence_length):
    module = _compressor(residual_scale=0.1)
    inputs = _inputs(sequence_length=sequence_length)
    fused_key, fused_value = module(
        (inputs[0], inputs[1]), (inputs[2], inputs[3])
    )
    assert fused_key.shape[2] == sequence_length
    assert fused_value.shape[2] == sequence_length


def test_latent_kv_compressor_state_dict_roundtrip():
    module = _compressor(residual_scale=0.1).eval()
    inputs = _inputs(sequence_length=8)
    expected = module((inputs[0], inputs[1]), (inputs[2], inputs[3]))

    restored = _compressor(residual_scale=0.1).eval()
    restored.load_state_dict(copy.deepcopy(module.state_dict()))
    actual = restored((inputs[0], inputs[1]), (inputs[2], inputs[3]))

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_latent_kv_projector_config_and_weights_roundtrip(tmp_path):
    module = _compressor(residual_scale=0.1).eval()
    config_path = tmp_path / "projector.json"
    weights_path = tmp_path / "projector.pt"
    save_projector(module, str(config_path))
    torch.save(module.state_dict(), weights_path)

    restored = load_projector(str(config_path))
    restored.load_state_dict(torch.load(weights_path, weights_only=True))

    assert isinstance(restored, LatentKVCompressor)
    assert restored.latent_dim == module.latent_dim
    assert torch.equal(restored.down_proj.weight, module.down_proj.weight)


def test_latent_kv_bridge_dynamic_cache_and_tuple_without_input_mutation():
    sharer_layers = [
        (
            torch.randn(1, 2, 4, 8),
            torch.randn(1, 2, 4, 8),
        )
        for _ in range(3)
    ]
    receiver_layers = [
        (
            torch.randn(1, 1, 4, 8),
            torch.randn(1, 1, 4, 8),
        )
        for _ in range(2)
    ]
    sharer_cache = _dynamic_cache(sharer_layers)
    receiver_cache = _dynamic_cache(receiver_layers)
    receiver_before = [
        (key.clone(), value.clone()) for key, value in receiver_layers
    ]
    bridge = LatentKVBridge(
        num_sharer_layers=3,
        num_receiver_layers=2,
        sharer_num_kv_heads=2,
        sharer_head_dim=8,
        receiver_num_kv_heads=1,
        receiver_head_dim=8,
        latent_dim=16,
        layer_mapping="proportional",
        init_residual_scale=0.1,
    )

    fused_dynamic = bridge(sharer_cache, receiver_cache)
    fused_tuple = bridge(tuple(sharer_layers), tuple(receiver_layers))

    assert isinstance(fused_dynamic, DynamicCache)
    assert isinstance(fused_tuple, tuple)
    assert len(fused_dynamic.key_cache) == 2
    assert bridge.layer_mapping == [0, 2]
    assert bridge.last_stats["fusion_type"] == "latent_kv_joint"
    for layer_idx, (key_before, value_before) in enumerate(receiver_before):
        assert torch.equal(receiver_cache.key_cache[layer_idx], key_before)
        assert torch.equal(receiver_cache.value_cache[layer_idx], value_before)


def test_proportional_mapping_validation():
    assert build_proportional_layer_mapping(5, 3) == [0, 2, 4]
    with pytest.raises(ValueError, match="positive"):
        build_proportional_layer_mapping(0, 3)


def test_latent_kv_rejects_sequence_length_mismatch_with_shapes():
    module = _compressor()
    inputs = _inputs(sequence_length=8)
    receiver_key = inputs[2][:, :, :-1]
    receiver_value = inputs[3][:, :, :-1]
    with pytest.raises(ValueError, match="equal batch and sequence lengths"):
        module((inputs[0], inputs[1]), (receiver_key, receiver_value))


def test_causal_generation_smoke_with_fused_dynamic_cache():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    sharer = LlamaForCausalLM(config).eval()
    receiver = LlamaForCausalLM(config).eval()
    sharer.requires_grad_(False)
    receiver.requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        sharer_output = sharer(prompt, use_cache=True)
        receiver_output = receiver(prompt, use_cache=True)

    bridge = LatentKVBridge(
        num_sharer_layers=2,
        num_receiver_layers=2,
        sharer_num_kv_heads=2,
        sharer_head_dim=8,
        receiver_num_kv_heads=2,
        receiver_head_dim=8,
        latent_dim=16,
        init_residual_scale=0.0,
    )
    fused_cache = bridge(
        sharer_output.past_key_values,
        receiver_output.past_key_values,
    )
    next_token = receiver_output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = []
    with torch.no_grad():
        for _ in range(4):
            generated.append(next_token)
            output = receiver(
                input_ids=next_token,
                past_key_values=fused_cache,
                use_cache=True,
            )
            fused_cache = output.past_key_values
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)

    assert torch.cat(generated, dim=1).shape == (1, 4)
