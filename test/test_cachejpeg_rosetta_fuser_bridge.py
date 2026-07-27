import torch
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge
from rosetta.model.latent_kv import LatentKVCompressor
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    resolve_adaptive_quant_table_config,
)


class IdentityProjector(torch.nn.Module):
    def forward(self, source_kv, target_kv):
        return source_kv


class DummyModel:
    class Config:
        num_hidden_layers = 2
        num_key_value_heads = 1
        num_attention_heads = 1
        hidden_size = 4
        head_dim = 4

    config = Config()


def _build_cache(layer_values):
    cache = DynamicCache()
    for value in layer_values:
        tensor = torch.tensor(value, dtype=torch.float32)
        cache.key_cache.append(tensor.clone())
        cache.value_cache.append((tensor + 10).clone())
    return cache


def test_fuser_bridge_projects_teacher_cache_into_base_layer():
    teacher_cache = _build_cache(
        [
            [[[[1.0, 2.0, 3.0, 4.0]]]],
            [[[[5.0, 6.0, 7.0, 8.0]]]],
        ]
    )
    base_cache = _build_cache(
        [
            [[[[0.0, 0.0, 0.0, 0.0]]]],
            [[[[9.0, 9.0, 9.0, 9.0]]]],
        ]
    )
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[IdentityProjector()],
        projector_dict={0: {1: {0: [(1, 0)]}}},
    )
    bridge = RosettaFuserBridge(assets)
    fused = bridge.fuse_teacher_cache_to_base(teacher_cache, base_seed_cache=base_cache)

    assert torch.equal(fused.key_cache[0], teacher_cache.key_cache[1])
    assert torch.equal(fused.value_cache[0], teacher_cache.value_cache[1])
    assert torch.equal(fused.key_cache[1], base_cache.key_cache[1])


def test_fuser_bridge_reports_latent_joint_fusion_stats():
    teacher_cache = _build_cache([[[[[1.0, 2.0, 3.0, 4.0]]]]])
    base_cache = _build_cache([[[[[0.0, 0.0, 0.0, 0.0]]]]])
    projector = LatentKVCompressor(
        sharer_num_kv_heads=1,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        latent_dim=8,
        init_residual_scale=0.1,
    )
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[projector],
        projector_dict={0: {1: {0: [(0, 0)]}}},
    )
    bridge = RosettaFuserBridge(assets)

    fused = bridge.fuse_teacher_cache_to_base(teacher_cache, base_seed_cache=base_cache)

    assert fused.key_cache[0].shape == base_cache.key_cache[0].shape
    assert bridge.last_fusion_stats["fusion_type"] == "latent_kv_joint"
    assert bridge.last_fusion_stats["latent_dim"] == 8
    assert len(bridge.last_fusion_stats["layers"]) == 1


def test_fuser_bridge_applies_trained_adaptive_quant_table_at_evaluation():
    teacher_cache = _build_cache(
        [
            [[[[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]]]],
            [[[[5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]]]],
        ]
    )
    base_cache = _build_cache(
        [
            [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
            [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
        ]
    )
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[IdentityProjector()],
        projector_dict={0: {1: {0: [(0, 0)], 1: [(1, 0)]}}},
    )
    config = resolve_adaptive_quant_table_config(
        {
            "enabled": True,
            "feature_bands": 2,
            "hidden_dim": 8,
            "alpha_candidates": [0.5, 1.0],
        }
    )
    quantizer = AdaptiveCoefficientQuantizer(
        num_layers=2, num_kv_heads=1, config=config
    ).eval()
    bridge = RosettaFuserBridge(assets, adaptive_quant_table=quantizer)

    fused = bridge.fuse_teacher_cache_to_base(
        teacher_cache, base_seed_cache=base_cache
    )

    assert fused.key_cache[0].shape == teacher_cache.key_cache[0].shape
    assert quantizer.last_result is not None
    assert quantizer.last_result.estimated_payload_bits.item() > 0
