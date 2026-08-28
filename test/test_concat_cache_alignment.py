import torch
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg_rosetta.cache_aligner import ConcatCacheAligner
from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets
from rosetta.cachejpeg_rosetta.pre_rope import (
    capture_pre_rope_keys,
    replace_cache_keys_with_pre_rope,
)
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    resolve_adaptive_quant_table_config,
)
from rosetta.model.latent_kv import LCFFirstProjector


class DummyConfig:
    num_hidden_layers = 2
    num_key_value_heads = 1
    num_attention_heads = 1
    hidden_size = 4
    head_dim = 4


class DummyModel(torch.nn.Module):
    config = DummyConfig()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class DummyPreRopeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 4
        self.k_proj = torch.nn.Identity()
        self.k_norm = None


class DummyPreRopeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = DummyPreRopeAttention()


class DummyPreRopeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([DummyPreRopeLayer()])
        self.model.rotary_emb = torch.nn.Identity()


def _cache(values):
    cache = DynamicCache()
    for value in values:
        tensor = torch.full((1, 1, 3, 4), float(value))
        cache.key_cache.append(tensor)
        cache.value_cache.append(tensor + 1)
    return cache


def test_replace_cache_keys_preserves_values_and_uses_captured_pre_rope_keys():
    post_rope = _cache([5, 6])
    captured = {
        0: torch.full((1, 1, 3, 4), 1.0),
        1: torch.full((1, 1, 3, 4), 2.0),
    }

    result = replace_cache_keys_with_pre_rope(post_rope, captured)

    assert torch.equal(result.key_cache[0], captured[0])
    assert torch.equal(result.key_cache[1], captured[1])
    assert torch.equal(result.value_cache[0], post_rope.value_cache[0])
    assert torch.equal(result.value_cache[1], post_rope.value_cache[1])


def test_pre_rope_capture_reuses_one_persistent_hook_set():
    model = DummyPreRopeModel()
    projection = model.model.layers[0].self_attn.k_proj

    with capture_pre_rope_keys(model) as first:
        projection(torch.ones(1, 3, 4))
    assert torch.equal(first[0], torch.ones(1, 1, 3, 4))
    assert len(projection._forward_hooks) == 1

    with capture_pre_rope_keys(model) as second:
        projection(torch.full((1, 3, 4), 2.0))
    assert torch.equal(second[0], torch.full((1, 1, 3, 4), 2.0))
    assert len(projection._forward_hooks) == 1


def test_concat_aligner_routes_and_projects_a_receiver_prefix(monkeypatch):
    projector = LCFFirstProjector(
        sharer_num_kv_heads=1,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        latent_dim=4,
    )
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[projector],
        projector_dict={0: {1: {0: [(1, 0)], 1: [(0, 0)]}}},
    )
    monkeypatch.setattr(
        "rosetta.cachejpeg_rosetta.cache_aligner.apply_receiver_compact_rope",
        lambda _model, cache: cache,
    )

    aligner = ConcatCacheAligner(assets)
    latent_cache, routing = aligner.encode(_cache([1, 2]))
    prefix = aligner.decode(latent_cache, routing)

    assert prefix.key_cache[0].shape == (1, 1, 3, 4)
    assert latent_cache[0][0].shape == (1, 1, 3, 2)
    assert routing.latent_dim == 4
    assert aligner.last_alignment_stats["codec_order"] == "lcf_down_cachejpeg_lcf_up"


def test_concat_aligner_streaming_layer_api(monkeypatch):
    projector = LCFFirstProjector(
        sharer_num_kv_heads=1,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        latent_dim=4,
    )
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[projector],
        projector_dict={0: {1: {0: [(1, 0)], 1: [(0, 0)]}}},
    )
    monkeypatch.setattr(
        "rosetta.cachejpeg_rosetta.cache_aligner.apply_receiver_compact_rope",
        lambda _model, cache: cache,
    )
    aligner = ConcatCacheAligner(assets)
    routing = aligner.prepare_routing()
    source = _cache([1, 2])
    decoded = {}
    for route in routing.routes:
        target_layer, source_layer, _projector_idx = route
        latent = aligner.encode_layer(
            route,
            source.key_cache[source_layer],
            source.value_cache[source_layer],
        )
        decoded[target_layer] = aligner.decode_layer(route, *latent)
    prefix = aligner.assemble_receiver_cache(decoded, routing)

    assert [route[1] for route in routing.routes] == [1, 0]
    assert prefix.key_cache[0].shape == (1, 1, 3, 4)


def test_lcf_first_qat_quantizes_latent_before_decode_and_keeps_gradients():
    projector = LCFFirstProjector(
        sharer_num_kv_heads=1,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=4,
        latent_dim=4,
    )
    quantizer = AdaptiveCoefficientQuantizer(
        num_layers=1,
        num_kv_heads=1,
        config=resolve_adaptive_quant_table_config(
            {"enabled": True, "feature_bands": 2, "hidden_dim": 8}
        ),
    )
    source = _cache([1])
    latent = projector.encode((source.key_cache[0], source.value_cache[0]))
    latent_key, latent_value = latent.chunk(2, dim=-1)
    quantized = quantizer(((latent_key.unsqueeze(1), latent_value.unsqueeze(1)),))
    reconstructed = torch.cat(
        (
            quantized.past_key_values[0][0].squeeze(1),
            quantized.past_key_values[0][1].squeeze(1),
        ),
        dim=-1,
    )
    receiver_key, receiver_value = projector.decode(reconstructed)

    (receiver_key.square().mean() + receiver_value.square().mean()).backward()

    assert reconstructed.shape == latent.shape
    assert quantized.estimated_payload_bits.item() > 0
    assert any(parameter.grad is not None for parameter in projector.parameters())
    assert any(parameter.grad is not None for parameter in quantizer.parameters())
