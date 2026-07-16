from script.evaluation.unified_evaluator import UnifiedEvaluator
from unittest.mock import patch


def _base_config(model_name="cachejpeg"):
    return {
        "model": {
            "model_name": model_name,
            "base_model_name": "Qwen/Qwen3-0.6B",
            "generation_config": {"max_new_tokens": 8, "do_sample": False},
        },
        "output": {"output_dir": "local/test_outputs/cachejpeg"},
        "eval": {
            "dataset": "longbench",
            "gpu_ids": [0],
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
        },
    }


def test_cachejpeg_model_type_detection():
    evaluator = UnifiedEvaluator(_base_config())
    assert evaluator.is_cachejpeg_model is True


def test_non_cachejpeg_model_type_detection():
    evaluator = UnifiedEvaluator(_base_config(model_name="Rosetta"))
    assert evaluator.is_cachejpeg_model is False


def test_cachejpeg_model_dispatch_uses_cachejpeg_loader():
    evaluator = UnifiedEvaluator(_base_config())
    with patch("script.evaluation.unified_evaluator.load_cachejpeg_model") as mocked_loader:
        mocked_loader.return_value = object(), object()
        with patch.object(evaluator, "evaluate_subject", return_value=([], 0.0, None, [], [])):
            evaluator.evaluate_on_gpu(rank=0, gpu_id=0, subjects=["qasper"], return_dict={})
    assert mocked_loader.called
