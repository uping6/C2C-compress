import json
import logging

import pytest
import torch

from rosetta.cachejpeg.transport import deserialize_payload, serialize_payload
from rosetta.cachejpeg_rosetta.adaptive_quant_codec import (
    AdaptiveQuantizedCachePayload,
    decode_adaptive_quantized_cache,
    encode_adaptive_quantized_cache,
)
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    _dct_ii_ortho,
    _idct_iii_ortho,
    resolve_adaptive_quant_table_config,
)
from script.train.SFT_train import (
    load_initial_projector_checkpoint,
    resolve_model_load_path,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "feature_bands": 4,
        "hidden_dim": 16,
        "alpha_candidates": [0.25, 0.5, 1.0],
        "initial_alpha_index": 1,
        "rate_weight": 1e-6,
    }
    values.update(overrides)
    return resolve_adaptive_quant_table_config(values)


def _cache(num_layers=2, batch=1, heads=2, sequence=9, head_dim=4):
    return tuple(
        (
            torch.randn(batch, heads, sequence, head_dim, requires_grad=True),
            torch.randn(batch, heads, sequence, head_dim, requires_grad=True),
        )
        for _ in range(num_layers)
    )


def test_fft_dct_round_trip_matches_input():
    values = torch.randn(2, 3, 11, 4)
    restored = _idct_iii_ortho(_dct_ii_ortho(values, axis=2), axis=2)
    assert torch.allclose(restored, values, atol=2e-5, rtol=2e-5)


def test_adaptive_quantizer_preserves_shapes_and_reports_tables():
    module = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=2, config=_config()
    ).eval()
    cache = _cache()
    result = module(cache)

    assert [pair[0].shape for pair in result.past_key_values] == [
        pair[0].shape for pair in cache
    ]
    assert result.alpha.shape == (1, 2, 2, 2)
    assert result.rounded_symbols.shape == (1, 2, 2, 2, 9, 4)
    assert result.estimated_payload_bits.item() > 0
    assert torch.equal(result.table_indices, torch.ones_like(result.table_indices))


def test_fixed_alpha_bypasses_allocator_and_reports_matching_table_index():
    torch.manual_seed(0)
    module = AdaptiveCoefficientQuantizer(
        num_layers=2,
        num_kv_heads=2,
        config=_config(fixed_alpha=1.0),
    ).eval()
    with torch.no_grad():
        module.alpha_head.bias[0] = 100.0
        module.alpha_head.bias[2] = -100.0

    result = module(_cache())

    assert torch.equal(result.alpha, torch.ones_like(result.alpha))
    assert torch.equal(result.table_indices, torch.full_like(result.table_indices, 2))


@pytest.mark.parametrize("fixed_alpha", [-1.0, 0.0, float("inf"), 0.75])
def test_fixed_alpha_must_be_positive_finite_candidate(fixed_alpha):
    with pytest.raises(ValueError, match="fixed_alpha"):
        _config(fixed_alpha=fixed_alpha)


def test_existing_state_dict_loads_strictly_with_fixed_alpha_override():
    original = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=1, config=_config()
    )
    overridden = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=1, config=_config(fixed_alpha=1.0)
    )

    overridden.load_state_dict(original.state_dict(), strict=True)


@pytest.mark.parametrize("representation", ["dense_int16", "adaptive_int"])
def test_adaptive_quant_wire_codec_round_trip(representation):
    torch.manual_seed(0)
    module = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=1, config=_config()
    ).eval()
    cache = _cache(heads=1)

    payload, result = encode_adaptive_quantized_cache(
        module,
        cache,
        representation=representation,
        backend="zlib1",
    )
    received = deserialize_payload(serialize_payload(payload))
    assert isinstance(received, AdaptiveQuantizedCachePayload)
    restored = decode_adaptive_quantized_cache(
        received,
        module,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for restored_pair, expected_pair in zip(restored, result.past_key_values):
        assert restored_pair[0].shape == expected_pair[0].shape
        assert restored_pair[1].shape == expected_pair[1].shape
        assert torch.allclose(restored_pair[0], expected_pair[0], atol=2e-3, rtol=2e-3)
        assert torch.allclose(restored_pair[1], expected_pair[1], atol=2e-3, rtol=2e-3)


def test_fixed_alpha_wire_codec_round_trip_uses_existing_candidate_index():
    torch.manual_seed(0)
    module = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=1, config=_config(fixed_alpha=1.0)
    ).eval()
    cache = _cache(heads=1)

    payload, result = encode_adaptive_quantized_cache(
        module,
        cache,
        representation="dense_int16",
        backend="zlib1",
    )
    restored = decode_adaptive_quantized_cache(
        deserialize_payload(serialize_payload(payload)),
        module,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(result.table_indices, torch.full_like(result.table_indices, 2))
    for restored_pair, expected_pair in zip(restored, result.past_key_values):
        assert torch.allclose(restored_pair[0], expected_pair[0], atol=2e-3, rtol=2e-3)
        assert torch.allclose(restored_pair[1], expected_pair[1], atol=2e-3, rtol=2e-3)


def test_adaptive_quantizer_backpropagates_task_and_rate_gradients():
    torch.manual_seed(0)
    module = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=2, config=_config()
    ).train()
    cache = _cache()
    result = module(cache)
    task_loss = sum(
        key.square().mean() + value.square().mean()
        for key, value in result.past_key_values
    )
    loss = task_loss + module.rate_weight(100) * result.estimated_payload_bits
    loss.backward()

    assert module.alpha_head.weight.grad is not None
    assert module.log_entropy_scales.grad is not None
    assert cache[0][0].grad is not None
    assert torch.isfinite(module.alpha_head.weight.grad).all()


def test_schedule_and_state_dict_round_trip():
    config = _config(initial_temperature=1.0, final_temperature=0.25, anneal_steps=10)
    module = AdaptiveCoefficientQuantizer(
        num_layers=1, num_kv_heads=1, config=config
    )
    assert module.update_temperature(5) == 0.5
    restored = AdaptiveCoefficientQuantizer(
        num_layers=1, num_kv_heads=1, config=config
    )
    restored.load_state_dict(module.state_dict())
    assert torch.equal(restored.alpha_head.bias, module.alpha_head.bias)
    assert restored.gumbel_temperature.item() == 0.5


def test_stage2_warm_start_loads_projectors_without_codec_or_optimizer(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    with torch.no_grad():
        source.weight.fill_(0.75)
        source.bias.fill_(-0.25)
        target.weight.zero_()
        target.bias.zero_()
    torch.save(source.state_dict(), tmp_path / "projector_0.pt")
    mapping = {0: {1: {0: [[0, 0]]}}}
    (tmp_path / "projector_config.json").write_text(
        json.dumps(mapping), encoding="utf-8"
    )

    holder = type("ProjectorHolder", (), {})()
    holder.projector_list = torch.nn.ModuleList([target])
    holder.projector_dict = mapping
    load_initial_projector_checkpoint(
        str(tmp_path), holder, "cpu", logging.getLogger("test")
    )

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_stage2_warm_start_accepts_json_list_for_runtime_tuple_mapping(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    torch.save(source.state_dict(), tmp_path / "projector_0.pt")
    (tmp_path / "projector_config.json").write_text(
        json.dumps({0: {1: {0: [(0, 0)]}}}), encoding="utf-8"
    )

    holder = type("ProjectorHolder", (), {})()
    holder.projector_list = torch.nn.ModuleList([target])
    holder.projector_dict = {0: {1: {0: [(0, 0)]}}}
    load_initial_projector_checkpoint(
        str(tmp_path), holder, "cpu", logging.getLogger("test")
    )

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)


def test_local_model_path_rejects_git_lfs_pointer(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:deadbeef\n"
        "size 3087467144\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        resolve_model_load_path(
            {
                "teacher_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "teacher_model_local_dir": str(tmp_path),
            },
            "teacher_model",
            "teacher_model_local_dir",
        )


def test_longbench_e_recipes_define_raw_then_qat_stages():
    with open(
        "recipe/train_recipe/C2C_longbench_latent_kv_split_raw_stage1.json",
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)
    with open(
        "recipe/train_recipe/C2C_longbench_latent_kv_split_adaptive_quant.json",
        "r",
        encoding="utf-8",
    ) as handle:
        qat = json.load(handle)

    assert raw["model"]["fusion_type"] == "latent_kv_split"
    assert qat["model"]["fusion_type"] == "latent_kv_split"
    assert raw["model"]["adaptive_quant_table"]["enabled"] is False
    assert raw["training"]["learning_rate"] == 3e-4
    assert qat["model"]["adaptive_quant_table"]["enabled"] is True
    assert qat["model"]["initial_projector_checkpoint"] == (
        raw["output"]["output_dir"] + "/final"
    )
    assert qat["training"]["learning_rate"] == 1e-4
    assert qat["model"]["adaptive_quant_table"]["rate_weight"] == 1e-6
