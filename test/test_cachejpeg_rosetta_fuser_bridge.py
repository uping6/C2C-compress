import torch
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge


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
