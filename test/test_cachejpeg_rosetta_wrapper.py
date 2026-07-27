import torch
from unittest.mock import patch

from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets
from rosetta.cachejpeg_rosetta.wrapper import CacheJPEGRosettaEvalWrapper, load_cachejpeg_rosetta_model


class DummyTokenizer:
    eos_token_id = 99


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
        self.last_input_ids = input_ids.clone()
        self.last_attention_mask = attention_mask.clone() if attention_mask is not None else None
        self.calls.append((self.last_input_ids, self.last_attention_mask))
        logits = torch.zeros((1, input_ids.shape[1], 8))
        return type("Out", (), {"logits": logits, "past_key_values": ()})()


class DummyCodec:
    def encode(self, cache, config):
        return {"cache": cache, "config": config}

    def decode(self, payload, config):
        return payload["cache"]


def test_cachejpeg_rosetta_wrapper_generate_returns_tensor():
    assets = LoadedRosettaAssets(
        base_model=DummyModel(),
        base_tokenizer=DummyTokenizer(),
        teacher_model=DummyModel(),
        teacher_tokenizer=DummyTokenizer(),
        projector_list=[],
        projector_dict={},
    )
    wrapper = CacheJPEGRosettaEvalWrapper.__new__(CacheJPEGRosettaEvalWrapper)
    wrapper.assets = assets
    wrapper.base_model = assets.base_model
    wrapper.base_tokenizer = assets.base_tokenizer
    wrapper.teacher_model = assets.teacher_model
    wrapper.teacher_tokenizer = assets.teacher_tokenizer
    wrapper.codec = DummyCodec()
    wrapper.codec_config = {}
    wrapper.fuser_bridge = type(
        "DummyBridge",
        (),
        {"fuse_teacher_cache_to_base": staticmethod(lambda cache, base_seed_cache=None: ())},
    )()
    wrapper._to_legacy_cache = lambda cache: cache
    wrapper._to_dynamic_cache = lambda cache: cache

    output = wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2


def test_cachejpeg_rosetta_wrapper_routes_aligned_ids_to_the_correct_models():
    base_model = DummyModel()
    teacher_model = DummyModel()
    assets = LoadedRosettaAssets(
        base_model=base_model,
        base_tokenizer=DummyTokenizer(),
        teacher_model=teacher_model,
        teacher_tokenizer=DummyTokenizer(),
        projector_list=[],
        projector_dict={},
    )
    wrapper = CacheJPEGRosettaEvalWrapper.__new__(CacheJPEGRosettaEvalWrapper)
    wrapper.assets = assets
    wrapper.base_model = base_model
    wrapper.base_tokenizer = assets.base_tokenizer
    wrapper.teacher_model = teacher_model
    wrapper.teacher_tokenizer = assets.teacher_tokenizer
    wrapper.codec = DummyCodec()
    wrapper.codec_config = {}
    wrapper.fuser_bridge = type(
        "DummyBridge",
        (),
        {"fuse_teacher_cache_to_base": staticmethod(lambda cache, base_seed_cache=None: ())},
    )()
    wrapper._to_legacy_cache = lambda cache: cache
    wrapper._to_dynamic_cache = lambda cache: cache

    base_ids = torch.tensor([[1, 2, 3]])
    teacher_ids = torch.tensor([[4, 5, 6]])
    base_mask = torch.tensor([[1, 1, 1]])
    teacher_mask = torch.tensor([[1, 0, 1]])
    output = wrapper.generate(
        input_ids=[base_ids, teacher_ids],
        attention_mask=[base_mask, teacher_mask],
        max_new_tokens=1,
        do_sample=False,
    )

    assert torch.equal(teacher_model.last_input_ids, teacher_ids)
    assert torch.equal(teacher_model.last_attention_mask, teacher_mask)
    assert torch.equal(base_model.calls[0][0], base_ids)
    assert torch.equal(base_model.calls[0][1], base_mask)
    # The base model is called again for decoding; the returned sequence must
    # still retain the receiver-tokenized prompt as its prefix.
    assert torch.equal(output[:, :base_ids.shape[1]], base_ids)


def test_receiver_only_loads_only_base_model():
    base_model = DummyModel()
    tokenizer = DummyTokenizer()
    model_config = {
        "rosetta_config": {
            "base_model": "receiver-model",
            "teacher_model": "sharer-model",
            "checkpoints_dir": "unused",
        },
        "cachejpeg_rosetta_config": {
            "ablation": {"receiver_only": True},
        },
    }

    with patch(
        "rosetta.cachejpeg_rosetta.wrapper.load_hf_model",
        return_value=(base_model, tokenizer),
    ) as load_base, patch(
        "rosetta.cachejpeg_rosetta.wrapper._load_rosetta_assets"
    ) as load_rosetta_assets:
        loaded_model, loaded_tokenizer = load_cachejpeg_rosetta_model(
            model_config,
            eval_config={},
            device=torch.device("cpu"),
            generation_config={"do_sample": False},
        )

    assert loaded_model is base_model
    assert loaded_tokenizer is tokenizer
    load_base.assert_called_once()
    load_rosetta_assets.assert_not_called()


def test_sharer_only_loads_only_teacher_model():
    teacher_model = DummyModel()
    tokenizer = DummyTokenizer()
    model_config = {
        "rosetta_config": {
            "base_model": "receiver-model",
            "teacher_model": "sharer-model",
            "checkpoints_dir": "unused",
        },
        "cachejpeg_rosetta_config": {
            "ablation": {"sharer_only": True},
        },
    }

    with patch(
        "rosetta.cachejpeg_rosetta.wrapper.load_hf_model",
        return_value=(teacher_model, tokenizer),
    ) as load_sharer, patch(
        "rosetta.cachejpeg_rosetta.wrapper._load_rosetta_assets"
    ) as load_rosetta_assets:
        loaded_model, loaded_tokenizer = load_cachejpeg_rosetta_model(
            model_config,
            eval_config={},
            device=torch.device("cpu"),
            generation_config={"do_sample": False},
        )

    assert loaded_model is teacher_model
    assert loaded_tokenizer is tokenizer
    assert load_sharer.call_args.args[0] == "sharer-model"
    load_rosetta_assets.assert_not_called()


def test_single_model_ablation_modes_are_mutually_exclusive():
    model_config = {
        "rosetta_config": {
            "base_model": "receiver-model",
            "teacher_model": "sharer-model",
        },
        "cachejpeg_rosetta_config": {
            "ablation": {"receiver_only": True, "sharer_only": True},
        },
    }

    try:
        load_cachejpeg_rosetta_model(
            model_config,
            eval_config={},
            device=torch.device("cpu"),
            generation_config={},
        )
    except ValueError as exc:
        assert "cannot both be enabled" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive ablation modes to fail")
