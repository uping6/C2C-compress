import hashlib

from script.evaluation.unified_evaluator import UnifiedEvaluator


def _heldout_rows(prefix, count):
    rows = []
    candidate = 0
    while len(rows) < count:
        example_id = f"{prefix}-{candidate}"
        digest = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16)
        if digest % 4 == 1:
            rows.append({"_id": example_id})
        candidate += 1
    return rows


def _evaluator_with_rows(rows_by_subject, seed=42):
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.longbench_e_test_subset_enabled = True
    evaluator.longbench_e_test_subset_size = 200
    evaluator.longbench_e_test_subset_seed = seed
    evaluator.longbench_e_test_subset_ids = {}
    evaluator.longbench_e_test_subset_counts = {}
    evaluator.dataset_config = {"test_split": "test"}
    evaluator._load_longbench_dataset = lambda subject: {
        "test": rows_by_subject[subject]
    }
    return evaluator


def test_longbench_e_subset_selects_exactly_200_heldout_samples_deterministically():
    rows = {
        "qasper_e": _heldout_rows("qasper", 160),
        "hotpotqa_e": _heldout_rows("hotpotqa", 160),
    }
    first = _evaluator_with_rows(rows, seed=17)
    second = _evaluator_with_rows(rows, seed=17)

    first._prepare_longbench_e_test_subset(list(rows))
    second._prepare_longbench_e_test_subset(list(rows))

    assert first.longbench_e_test_subset_ids == second.longbench_e_test_subset_ids
    assert sum(first.longbench_e_test_subset_counts.values()) == 200
    for selected_ids in first.longbench_e_test_subset_ids.values():
        for example_id in selected_ids:
            digest = int(
                hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16
            )
            assert digest % 4 == 1


def test_longbench_e_subset_seed_changes_selection_and_uses_separate_output_dir():
    rows = {
        "qasper_e": _heldout_rows("qasper", 160),
        "hotpotqa_e": _heldout_rows("hotpotqa", 160),
    }
    first = _evaluator_with_rows(rows, seed=1)
    second = _evaluator_with_rows(rows, seed=2)
    first._prepare_longbench_e_test_subset(list(rows))
    second._prepare_longbench_e_test_subset(list(rows))

    assert first.longbench_e_test_subset_ids != second.longbench_e_test_subset_ids
    assert first._longbench_prediction_split_dir(True) == "pred_e_subset_200_seed_1"
    assert first._longbench_prediction_split_dir(False) == "pred"
