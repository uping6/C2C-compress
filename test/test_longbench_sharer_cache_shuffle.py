import torch

from script.evaluation.unified_evaluator import UnifiedEvaluator


def test_sharer_shuffle_pairs_are_deterministic_and_have_no_self_matches():
    indices = [2, 5, 9, 12]
    first = UnifiedEvaluator._build_sharer_shuffle_pairs(
        indices, seed=17, subject="qasper_e"
    )
    second = UnifiedEvaluator._build_sharer_shuffle_pairs(
        indices, seed=17, subject="qasper_e"
    )

    assert first == second
    assert set(first) == set(indices)
    assert set(first.values()) == set(indices)
    assert all(receiver != sharer for receiver, sharer in first.items())


def test_compose_shuffled_sharer_inputs_left_pads_shorter_teacher():
    receiver = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    sharer = {
        "input_ids": torch.tensor([[8, 9]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    inputs, metadata = UnifiedEvaluator._compose_shuffled_sharer_inputs(
        receiver, sharer, teacher_pad_token_id=0
    )

    assert torch.equal(inputs["input_ids"][0], receiver["input_ids"])
    assert torch.equal(inputs["input_ids"][1], torch.tensor([[0, 0, 8, 9]]))
    assert torch.equal(inputs["attention_mask"][1], torch.tensor([[0, 0, 1, 1]]))
    assert metadata == {
        "receiver_cache_length": 4,
        "sharer_original_length": 2,
        "sharer_used_length": 4,
    }


def test_compose_shuffled_sharer_inputs_keeps_both_ends_when_cropping():
    receiver = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }
    sharer = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15]]),
        "attention_mask": torch.ones((1, 6), dtype=torch.long),
    }

    inputs, metadata = UnifiedEvaluator._compose_shuffled_sharer_inputs(
        receiver, sharer, teacher_pad_token_id=0
    )

    assert torch.equal(inputs["input_ids"][1], torch.tensor([[10, 11, 14, 15]]))
    assert metadata["sharer_original_length"] == 6
    assert metadata["sharer_used_length"] == 4
