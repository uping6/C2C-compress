import torch

from rosetta.train.dataset_adapters import (
    IndependentDualTokenizerChatDataset,
    IndependentDualTokenizerCollator,
)


class _ChatSource:
    def __init__(self):
        self.samples = [[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class _Tokenizer:
    def __init__(self, offset, pad_token_id):
        self.offset = offset
        self.pad_token_id = pad_token_id

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **_kwargs,
    ):
        assert tokenize is False
        rendered = "|".join(message["content"] for message in messages)
        return rendered + ("|GEN" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Different offsets and token grouping make the two token streams
        # observably independent rather than position-aligned copies.
        width = 2 if self.offset else 1
        return {
            "input_ids": [
                self.offset + len(text[index : index + width])
                for index in range(0, len(text), width)
            ]
        }


def test_independent_dataset_uses_sharer_tokenizer_for_prompt_only():
    receiver = _Tokenizer(offset=0, pad_token_id=0)
    sharer = _Tokenizer(offset=100, pad_token_id=99)
    dataset = IndependentDualTokenizerChatDataset(
        _ChatSource(), receiver, sharer, max_length=128
    )

    sample = dataset[0]

    assert sample["sharer_prompt_only"] is True
    assert len(sample["input_ids"][0]) != len(sample["input_ids"][1])
    assert all(token >= 100 for token in sample["input_ids"][1])
    assert any(label != -100 for label in sample["labels"])
    assert len(sample["labels"]) == len(sample["input_ids"][0])


def test_independent_collator_pads_model_streams_to_separate_lengths():
    receiver = _Tokenizer(offset=0, pad_token_id=0)
    sharer = _Tokenizer(offset=100, pad_token_id=99)
    collator = IndependentDualTokenizerCollator(
        receiver, sharer, max_length=16
    )
    features = [
        {
            "input_ids": [[1, 2, 3, 4], [101, 102]],
            "labels": [-100, -100, 3, 4],
            "sharer_prompt_only": True,
        },
        {
            "input_ids": [[5, 6, 7], [103, 104, 105]],
            "labels": [-100, 6, 7],
            "sharer_prompt_only": True,
        },
    ]

    batch = collator(features)

    assert batch["input_ids"][0].shape == (2, 4)
    assert batch["input_ids"][1].shape == (2, 3)
    assert batch["input_ids"][0][1, -1].item() == 0
    assert batch["input_ids"][1][0, -1].item() == 99
    assert torch.equal(batch["attention_mask"][1][0], torch.tensor([1.0, 1.0, 0.0]))
    assert batch["labels"][1, -1].item() == -100
    assert batch["sharer_prompt_only"] is True

