import torch

from rosetta.cachejpeg.wrapper import CacheJPEGEvalWrapper


class DummyTokenizer:
    eos_token_id = 99

    def __call__(self, text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        logits = torch.zeros((1, input_ids.shape[1], 8))
        return type("Out", (), {"logits": logits, "past_key_values": ()})()


def test_cachejpeg_wrapper_generate_returns_tensor():
    wrapper = CacheJPEGEvalWrapper(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        codec_config={"method": "cachejpeg"},
    )
    output = wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
