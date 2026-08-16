from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg.transport import DirectTransport
from rosetta.cachejpeg_rosetta.fuser_bridge import (
    LoadedRosettaAssets,
    RosettaFuserBridge,
)
from rosetta.cachejpeg_rosetta.wrapper import CacheJPEGRosettaEvalWrapper
from rosetta.model.adaptive_quant_table import (
    AdaptiveCoefficientQuantizer,
    resolve_adaptive_quant_table_config,
)
from rosetta.model.latent_kv import (
    LatentKVPayload,
    ReceiverKVDecoder,
    SharerKVEncoder,
    SplitLatentKVProjector,
)


class DummyModel:
    class Config:
        num_hidden_layers = 2
        num_key_value_heads = 1
        num_attention_heads = 1
        hidden_size = 8
        head_dim = 8

    config = Config()


class TinyCacheModel(torch.nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = type("Config", (), {"num_hidden_layers": 1})()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=True,
        past_key_values=None,
    ):
        if past_key_values is None:
            batch, sequence_length = input_ids.shape
            key = torch.ones(
                batch, self.num_heads, sequence_length, self.head_dim
            )
            value = key + 1
            past_key_values = _cache([(key, value)])
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 16)
        logits[..., 3] = 1
        return type(
            "Output",
            (),
            {"past_key_values": past_key_values, "logits": logits},
        )()


class TinyTokenizer:
    eos_token_id = None


def _cache(values):
    cache = DynamicCache()
    for key, value in values:
        cache.key_cache.append(key)
        cache.value_cache.append(value)
    return cache


def _projector(residual_scale=0.0):
    return SplitLatentKVProjector(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=8,
        latent_dim=8,
        mlp_ratio=2.0,
        init_residual_scale=residual_scale,
    )


def test_sharer_encoder_has_no_receiver_input_and_returns_compressed_latent():
    encoder = SharerKVEncoder(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        latent_dim=8,
        mlp_ratio=2.0,
    )
    sharer_key = torch.randn(2, 2, 5, 4)
    sharer_value = torch.randn(2, 2, 5, 4)

    latent = encoder(sharer_key, sharer_value)

    assert latent.shape == (2, 5, 8)
    assert encoder.input_dim == 16


def test_receiver_decoder_identity_initialization_and_shape():
    decoder = ReceiverKVDecoder(
        receiver_num_kv_heads=2,
        receiver_head_dim=4,
        latent_dim=8,
        init_residual_scale=0.0,
    )
    latent = torch.randn(2, 5, 8)
    receiver_key = torch.randn(2, 2, 5, 4)
    receiver_value = torch.randn(2, 2, 5, 4)

    fused_key, fused_value = decoder(latent, receiver_key, receiver_value)

    assert fused_key.shape == receiver_key.shape
    assert fused_value.shape == receiver_value.shape
    assert torch.equal(fused_key, receiver_key)
    assert torch.equal(fused_value, receiver_value)


def test_split_projector_propagates_gradients_through_both_sides():
    projector = _projector(residual_scale=0.1)
    sharer_key = torch.randn(2, 2, 5, 4)
    sharer_value = torch.randn(2, 2, 5, 4)
    receiver_key = torch.randn(2, 1, 5, 8)
    receiver_value = torch.randn(2, 1, 5, 8)

    outputs = projector(
        (sharer_key, sharer_value), (receiver_key, receiver_value)
    )
    sum(output.square().mean() for output in outputs).backward()

    assert projector.encoder.down_proj.weight.grad is not None
    assert projector.decoder.receiver_condition_proj.weight.grad is not None
    assert projector.decoder.key_up_proj.weight.grad is not None
    assert projector.decoder.value_up_proj.weight.grad is not None


def test_split_bridge_transports_only_cpu_latents_and_decodes_on_receiver():
    projectors = [_projector(residual_scale=0.1) for _ in range(2)]
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=projectors,
        projector_dict={
            0: {
                1: {
                    0: [(0, 0)],
                    1: [(1, 1)],
                }
            }
        },
    )
    bridge = RosettaFuserBridge(assets)
    sharer_cache = _cache(
        [
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
        ]
    )
    receiver_cache = _cache(
        [
            (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),
            (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),
        ]
    )

    payload = bridge.encode_teacher_cache_to_latents(sharer_cache)
    received_payload = DirectTransport().roundtrip(payload)
    fused = bridge.fuse_latents_to_base(received_payload, receiver_cache)

    assert isinstance(payload, LatentKVPayload)
    assert payload.quantized is False
    assert len(payload.layers) == 2
    assert all(layer.latent.device.type == "cpu" for layer in payload.layers)
    assert all(layer.latent.shape == (1, 4, 8) for layer in payload.layers)
    assert isinstance(fused, DynamicCache)
    assert fused.key_cache[0].shape == receiver_cache.key_cache[0].shape
    assert bridge.last_fusion_stats["fusion_type"] == "latent_kv_split"


def test_transmitted_latent_is_independent_of_receiver_cache():
    projector = _projector(residual_scale=0.1)
    sharer_key = torch.randn(1, 2, 4, 4)
    sharer_value = torch.randn(1, 2, 4, 4)
    receiver_a = (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8))
    receiver_b = (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8))

    latent = projector.encode((sharer_key, sharer_value))
    latent_before = latent.clone()
    output_a = projector.decode(latent, receiver_a)
    output_b = projector.decode(latent, receiver_b)

    assert torch.equal(latent, latent_before)
    assert not torch.equal(output_a[0], output_b[0])


def test_split_bridge_applies_adaptive_quantization_before_latent_encoding():
    projectors = [_projector(residual_scale=0.1) for _ in range(2)]
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=projectors,
        projector_dict={0: {1: {0: [(0, 0)], 1: [(1, 1)]}}},
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
        num_layers=2, num_kv_heads=2, config=config
    ).eval()
    bridge = RosettaFuserBridge(assets, adaptive_quant_table=quantizer)
    sharer_cache = _cache(
        [
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
            (torch.randn(1, 2, 4, 4), torch.randn(1, 2, 4, 4)),
        ]
    )
    receiver_cache = _cache(
        [
            (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),
            (torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)),
        ]
    )

    payload = bridge.encode_teacher_cache_to_latents(sharer_cache)
    bridge.fuse_latents_to_base(payload, receiver_cache)

    assert quantizer.last_result is not None
    assert quantizer.last_result.estimated_payload_bits.item() > 0
    assert bridge.last_fusion_stats["adaptive_quant_table"][
        "estimated_payload_bits"
    ] > 0


def test_split_wrapper_runs_encode_transport_decode_generation_path():
    base_model = TinyCacheModel(num_heads=1, head_dim=8)
    teacher_model = TinyCacheModel(num_heads=2, head_dim=4)
    projector = _projector(residual_scale=0.1)
    assets = LoadedRosettaAssets(
        base_model=base_model,
        base_tokenizer=TinyTokenizer(),
        teacher_model=teacher_model,
        teacher_tokenizer=TinyTokenizer(),
        projector_list=[projector],
        projector_dict={0: {1: {0: [(0, 0)]}}},
    )
    wrapper = CacheJPEGRosettaEvalWrapper.__new__(CacheJPEGRosettaEvalWrapper)
    wrapper.assets = assets
    wrapper.base_model = base_model
    wrapper.base_tokenizer = assets.base_tokenizer
    wrapper.teacher_model = teacher_model
    wrapper.teacher_tokenizer = assets.teacher_tokenizer
    wrapper.fuser_bridge = RosettaFuserBridge(assets)
    wrapper.fusion_type = "latent_kv_split"
    wrapper.transport = DirectTransport()
    wrapper.last_transport_stats = None
    wrapper.last_codec_stats = None
    wrapper.last_fusion_stats = None
    wrapper._to_legacy_cache = lambda cache: tuple(
        zip(cache.key_cache, cache.value_cache)
    )
    wrapper._to_dynamic_cache = lambda cache: cache

    output = wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
    )

    assert output.shape == (1, 5)
    assert wrapper.last_codec_stats["mode"] == "latent_kv_split"
    assert wrapper.last_codec_stats["latent_bytes"] > 0
    assert wrapper.last_transport_stats.payload_bytes > 0
    assert wrapper.last_fusion_stats["fusion_type"] == "latent_kv_split"


def test_split_wrapper_initialization_does_not_load_cachejpeg_codec():
    assets = LoadedRosettaAssets(
        base_model=TinyCacheModel(num_heads=1, head_dim=8),
        base_tokenizer=TinyTokenizer(),
        teacher_model=TinyCacheModel(num_heads=2, head_dim=4),
        teacher_tokenizer=TinyTokenizer(),
        projector_list=[_projector()],
        projector_dict={0: {1: {0: [(0, 0)]}}},
    )
    config = {
        "fusion_type": "latent_kv_split",
        "latent_kv_bridge": {"enabled": True, "latent_dim": 8},
        "transport": {"mode": "direct"},
    }

    with patch(
        "rosetta.cachejpeg_rosetta.wrapper._ensure_homo_imports",
        side_effect=AssertionError("split mode must not load CacheJPEG"),
    ):
        wrapper = CacheJPEGRosettaEvalWrapper(assets, config)

    assert wrapper.codec is None
    assert wrapper.fusion_type == "latent_kv_split"


def test_split_wrapper_cachejpeg_zlib_compresses_latent_before_transport():
    assets = LoadedRosettaAssets(
        base_model=TinyCacheModel(num_heads=1, head_dim=8),
        base_tokenizer=TinyTokenizer(),
        teacher_model=TinyCacheModel(num_heads=2, head_dim=4),
        teacher_tokenizer=TinyTokenizer(),
        projector_list=[_projector(residual_scale=0.1)],
        projector_dict={0: {1: {0: [(0, 0)]}}},
    )
    config = {
        "fusion_type": "latent_kv_split",
        "latent_kv_bridge": {"enabled": True, "latent_dim": 8},
        "split_latent_cachejpeg": {
            "enabled": True,
            "codec": {
                "anchors": {"sink_count": 1},
                "block": {"mode": "global"},
                "quant": {"low": 1.0, "high": 8.0},
                "compute": {"backend": "gpu", "transform_dtype": "float32"},
                "entropy": {
                    "representation": "dense_int16",
                    "backend": "zlib1",
                },
            },
        },
        "transport": {"mode": "direct"},
    }
    wrapper = CacheJPEGRosettaEvalWrapper(assets, config)

    output = wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        max_new_tokens=2,
        do_sample=False,
    )

    assert output.shape == (1, 5)
    assert wrapper.last_codec_stats["mode"] == "latent_kv_split_cachejpeg"
    assert wrapper.last_codec_stats["quantized"] is True
    assert wrapper.last_codec_stats["entropy_backend"] == "zlib1"
    assert wrapper.last_codec_stats["payload_bytes"] > 0
    assert wrapper.last_codec_stats["cachejpeg_encode_seconds"] >= 0
    assert wrapper.last_codec_stats["cachejpeg_decode_seconds"] >= 0
    assert wrapper.last_transport_stats.payload_bytes > 0
    assert wrapper.last_fusion_stats["fusion_type"] == "latent_kv_split"
