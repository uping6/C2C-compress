from unittest.mock import patch

import pytest
import torch

from rosetta.cachejpeg_rosetta.direct_mlp_cache_aligner import (
    DirectMLPConcatCacheAligner,
)
from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets
from rosetta.cachejpeg_rosetta.wrapper import CacheJPEGRosettaEvalWrapper
from rosetta.model.direct_pre_rope_mlp import DirectPreRopeMLPProjector
from rosetta.model.projector import create_projector, load_projector, save_projector


class DummyConfig:
    num_hidden_layers = 2


class DummyModel(torch.nn.Module):
    config = DummyConfig()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def _projector():
    return DirectPreRopeMLPProjector(
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=6,
        hidden_dim=8,
    )


def _assets():
    return LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=None,
        teacher_model=DummyModel(),
        teacher_tokenizer=None,
        projector_list=[_projector(), _projector()],
        projector_dict={0: {1: {0: [(1, 0)], 1: [(0, 1)]}}},
    )


def _source_cache():
    return tuple(
        (
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        )
        for _ in range(2)
    )


def test_direct_projector_aligns_geometry_and_preserves_sequence_length():
    projector = _projector()
    key, value = projector.project(_source_cache()[0])

    assert key.shape == value.shape == (1, 1, 3, 6)
    assert projector.key_mlp[0].weight.data_ptr() != projector.value_mlp[0].weight.data_ptr()


def test_direct_projector_is_registered_and_both_mlps_receive_gradients():
    projector = create_projector(
        "DirectPreRopeMLPProjector",
        sharer_num_kv_heads=2,
        sharer_head_dim=4,
        receiver_num_kv_heads=1,
        receiver_head_dim=6,
        hidden_dim=8,
    )
    key, value = projector(_source_cache()[0])
    (key.square().mean() + value.square().mean()).backward()

    assert projector.key_mlp[0].weight.grad is not None
    assert projector.value_mlp[0].weight.grad is not None


def test_direct_projector_config_roundtrip(tmp_path):
    projector = _projector()
    config_path = tmp_path / "projector.json"
    save_projector(projector, str(config_path))

    restored = load_projector(str(config_path))

    assert isinstance(restored, DirectPreRopeMLPProjector)
    assert restored.hidden_dim == projector.hidden_dim
    assert restored.receiver_head_dim == projector.receiver_head_dim


def test_direct_aligner_routes_layers_and_applies_receiver_rope(monkeypatch):
    aligner = DirectMLPConcatCacheAligner(_assets())
    calls = []

    def fake_rope(_model, cache):
        calls.append(cache)
        return cache

    monkeypatch.setattr(
        "rosetta.cachejpeg_rosetta.direct_mlp_cache_aligner.apply_receiver_compact_rope",
        fake_rope,
    )
    prefix = aligner.align(_source_cache())

    assert [route[1] for route in aligner.prepare_routing()] == [1, 0]
    assert len(prefix.key_cache) == 2
    assert prefix.key_cache[0].shape == (1, 1, 3, 6)
    assert len(calls) == 1
    assert aligner.last_alignment_stats["communication_mode"] == "local_direct"


def test_direct_wrapper_initialization_skips_codec_transport_and_fuser():
    with patch(
        "rosetta.cachejpeg_rosetta.wrapper._ensure_homo_imports",
        side_effect=AssertionError("direct concat must not initialize CacheJPEG"),
    ), patch(
        "rosetta.cachejpeg_rosetta.wrapper.build_transport",
        side_effect=AssertionError("direct concat must not initialize transport"),
    ), patch(
        "rosetta.cachejpeg_rosetta.wrapper.RosettaFuserBridge",
        side_effect=AssertionError("direct concat must not initialize a fuser"),
    ):
        wrapper = CacheJPEGRosettaEvalWrapper(
            assets=_assets(),
            codec_config={
                "cache_alignment": "concat",
                "fusion_type": "original",
                "concat_projector": {"type": "direct_pre_rope_mlp"},
                "adaptive_quant_table": {"enabled": False},
                "layer_streaming": {"enabled": False},
            },
        )

    assert wrapper.codec is None
    assert wrapper.transport is None
    assert wrapper.fuser_bridge is None
    assert isinstance(wrapper.concat_cache_aligner, DirectMLPConcatCacheAligner)


@pytest.mark.parametrize(
    "invalid_section",
    [
        {"adaptive_quant_table": {"enabled": True}},
        {"layer_streaming": {"enabled": True}},
    ],
)
def test_direct_wrapper_rejects_quantization_and_streaming(invalid_section):
    config = {
        "cache_alignment": "concat",
        "fusion_type": "original",
        "concat_projector": {"type": "direct_pre_rope_mlp"},
        **invalid_section,
    }
    with pytest.raises(ValueError, match="direct_pre_rope_mlp"):
        CacheJPEGRosettaEvalWrapper(assets=_assets(), codec_config=config)
