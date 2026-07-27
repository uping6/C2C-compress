from rosetta.cachejpeg_rosetta.config import resolve_cachejpeg_rosetta_eval_config


def test_resolve_cachejpeg_rosetta_eval_config_reads_nested_values():
    cfg = resolve_cachejpeg_rosetta_eval_config(
        {
            "sharer_model_role": "teacher",
            "receiver_model_role": "base",
            "homo_c2c_kv_src": "/tmp/homo",
            "layer_streaming": {"enabled": True, "queue_size": 3},
            "codec": {
                "method": "cachejpeg",
                "anchors": {"sink_count": 2},
                "block": {"mode": "global", "size": 64},
            },
        }
    )
    assert cfg.sharer_model_role == "teacher"
    assert cfg.receiver_model_role == "base"
    assert cfg.homo_c2c_kv_src == "/tmp/homo"
    assert cfg.codec.anchors.sink_count == 2
    assert cfg.layer_streaming.enabled is True
    assert cfg.layer_streaming.queue_size == 3
