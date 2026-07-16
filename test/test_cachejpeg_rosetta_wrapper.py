import torch

from rosetta.cachejpeg_rosetta.fuser_bridge import LoadedRosettaAssets
from rosetta.cachejpeg_rosetta.wrapper import CacheJPEGRosettaEvalWrapper


class DummyTokenizer:
    eos_token_id = 99


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=True):
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
