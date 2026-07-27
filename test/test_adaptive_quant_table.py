import json
import logging

import torch

from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    _dct_ii_ortho,
    _idct_iii_ortho,
    resolve_adaptive_quant_table_config,
)
from script.train.SFT_train import load_initial_projector_checkpoint


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


def test_longbench_e_recipes_define_raw_then_qat_stages():
    with open(
        "recipe/train_recipe/C2C_longbench_latent_kv_joint_raw_stage1.json",
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)
    with open(
        "recipe/train_recipe/C2C_longbench_latent_kv_joint_adaptive_quant.json",
        "r",
        encoding="utf-8",
    ) as handle:
        qat = json.load(handle)

    assert raw["model"]["adaptive_quant_table"]["enabled"] is False
    assert raw["training"]["learning_rate"] == 3e-4
    assert qat["model"]["adaptive_quant_table"]["enabled"] is True
    assert qat["model"]["initial_projector_checkpoint"] == (
        raw["output"]["output_dir"] + "/final"
    )
    assert qat["training"]["learning_rate"] == 1e-4
    assert qat["model"]["adaptive_quant_table"]["rate_weight"] == 1e-6
