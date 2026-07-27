import json

from script.evaluation.unified_evaluator import UnifiedEvaluator


def test_longbench_summary_aggregates_streaming_latency_and_layer_timings(tmp_path):
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.dataset_name = "longbench"
    evaluator.model_config = {"model_name": "cachejpeg_rosetta"}
    evaluator.eval_config = {"answer_method": "generate"}
    evaluator.output_dir = tmp_path
    logs = [
        {
            "answer_latency_ms": 100.0,
            "cachejpeg_stats": {
                "layer_encode_seconds": [0.01, 0.03],
                "layer_prefill_seconds": [0.002, 0.004],
                "pipeline_seconds": 0.08,
                "payload_bytes": 1000,
            },
        },
        {
            "answer_latency_ms": 200.0,
            "cachejpeg_stats": {
                "layer_encode_seconds": [0.02, 0.04],
                "layer_prefill_seconds": [0.004, 0.006],
                "pipeline_seconds": 0.12,
                "payload_bytes": 2000,
            },
        },
    ]

    evaluator.save_results(
        [], {}, {}, {}, [], logs,
        longbench_subject_metrics=[
            {"subject": "qasper_e", "metric": "f1", "score": 0.5, "num_samples": 2}
        ],
    )

    performance_file = next(tmp_path.glob("*_performance.json"))
    performance = json.loads(performance_file.read_text())
    assert performance["end_to_end_avg_ms"] == 150.0
    assert performance["longbench_e_score"] == 0.5
    assert performance["avg_layer_encode_ms"] == 25.0
    assert performance["avg_layer_prefill_ms"] == 4.0
    assert performance["per_layer_avg_encode_ms"] == [15.0, 35.0]


def test_longbench_summary_aggregates_non_streaming_codec_timings(tmp_path):
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.dataset_name = "longbench"
    evaluator.model_config = {"model_name": "cachejpeg_rosetta"}
    evaluator.eval_config = {"answer_method": "generate"}
    evaluator.output_dir = tmp_path

    evaluator.save_results(
        [], {}, {}, {}, [],
        [{
            "answer_latency_ms": 250.0,
            "cachejpeg_stats": {
                "encode_seconds": 0.02,
                "decode_seconds": 0.03,
                "payload_bytes": 2048,
            },
        }],
        longbench_subject_metrics=[],
    )

    performance_file = next(tmp_path.glob("*_performance.json"))
    performance = json.loads(performance_file.read_text())
    assert performance["avg_encode_ms"] == 20.0
    assert performance["avg_decode_ms"] == 30.0
    assert performance["avg_payload_bytes"] == 2048.0
    assert "avg_layer_encode_ms" not in performance
    assert "avg_pipeline_ms" not in performance


def test_longbench_summary_omits_codec_timings_for_baseline(tmp_path):
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.dataset_name = "longbench"
    evaluator.model_config = {"model_name": "Rosetta"}
    evaluator.eval_config = {"answer_method": "generate"}
    evaluator.output_dir = tmp_path

    evaluator.save_results(
        [], {}, {}, {}, [],
        [{"answer_latency_ms": 100.0}],
        longbench_subject_metrics=[],
    )

    performance_file = next(tmp_path.glob("*_performance.json"))
    performance = json.loads(performance_file.read_text())
    assert performance["end_to_end_avg_ms"] == 100.0
    assert "avg_encode_ms" not in performance
    assert "avg_layer_encode_ms" not in performance
    assert "avg_payload_bytes" not in performance


def test_longbench_length_statistics_accept_none_correctness_from_parallel_workers():
    evaluator = UnifiedEvaluator.__new__(UnifiedEvaluator)
    evaluator.dataset_name = "longbench"
    stats = [
        {
            "subject": "qasper_e",
            "input_length": 100,
            "gen_length": 10,
            "length_ratio": 0.1,
            "is_correct": None,
        },
        {
            "subject": "qasper_e",
            "input_length": 200,
            "gen_length": 20,
            "length_ratio": 0.1,
            "is_correct": None,
        },
    ]

    summary = evaluator._compute_length_statistics(stats)

    assert summary["subjects"]["qasper_e"]["accuracy"] is None
    assert summary["subjects"]["qasper_e"]["avg_input_length"] == 150.0
