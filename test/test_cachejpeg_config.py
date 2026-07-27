from rosetta.cachejpeg.config import resolve_cachejpeg_eval_config


def test_resolve_cachejpeg_eval_config_reads_nested_values():
    cfg = resolve_cachejpeg_eval_config(
        {
            "method": "cachejpeg",
            "anchors": {"sink_count": 2},
            "block": {"mode": "global", "size": 64},
            "quant": {"low": 1.0, "high": 8.0, "curve": "quadratic"},
            "entropy": {"representation": "dense_int16", "backend": "zlib1"},
        }
    )
    assert cfg.method == "cachejpeg"
    assert cfg.anchors.sink_count == 2
    assert cfg.block.mode == "global"
    assert cfg.entropy.backend == "zlib1"


def test_resolve_cachejpeg_eval_config_accepts_adaptive_int_representation():
    cfg = resolve_cachejpeg_eval_config(
        {"entropy": {"representation": "adaptive_int", "backend": "lz4"}}
    )
    assert cfg.entropy.representation == "adaptive_int"


def test_resolve_cachejpeg_eval_config_defaults_transport_to_none():
    cfg = resolve_cachejpeg_eval_config({})
    assert cfg.transport.mode == "none"
