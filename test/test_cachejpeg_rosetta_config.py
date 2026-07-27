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
    assert cfg.fusion_type == "original"
    assert cfg.latent_kv_bridge.enabled is False


def test_resolve_cachejpeg_rosetta_eval_config_reads_latent_fusion():
    cfg = resolve_cachejpeg_rosetta_eval_config(
        {
            "fusion_type": "latent_kv_joint",
            "latent_kv_bridge": {
                "enabled": True,
                "latent_dim": 64,
                "layer_mapping": [0, 2],
            },
        }
    )
    assert cfg.fusion_type == "latent_kv_joint"
    assert cfg.latent_kv_bridge.enabled is True
    assert cfg.latent_kv_bridge.latent_dim == 64
    assert cfg.latent_kv_bridge.layer_mapping == (0, 2)


def test_resolve_cachejpeg_rosetta_eval_config_reads_adaptive_quant_table():
    cfg = resolve_cachejpeg_rosetta_eval_config(
        {
            "adaptive_quant_table": {
                "enabled": True,
                "alpha_candidates": [0.25, 1.0, 4.0],
                "rate_weight": 1e-6,
            }
        }
    )
    assert cfg.adaptive_quant_table.enabled is True
    assert cfg.adaptive_quant_table.alpha_candidates == (0.25, 1.0, 4.0)
    assert cfg.adaptive_quant_table.rate_weight == 1e-6


def test_resolve_cachejpeg_rosetta_eval_config_accepts_split_mode():
    cfg = resolve_cachejpeg_rosetta_eval_config(
        {"fusion_type": "latent_kv_split"}
    )
    assert cfg.fusion_type == "latent_kv_split"
    assert cfg.latent_kv_bridge.enabled is True
    assert cfg.split_latent_cachejpeg.enabled is False


def test_resolve_cachejpeg_rosetta_eval_config_rejects_split_streaming():
    try:
        resolve_cachejpeg_rosetta_eval_config(
            {
                "fusion_type": "latent_kv_split",
                "layer_streaming": {"enabled": True},
            }
        )
    except ValueError as exc:
        assert "layer_streaming" in str(exc)
    else:
        raise AssertionError("Expected split mode to reject layer streaming")


def test_resolve_split_latent_cachejpeg_requires_zlib_and_split_mode():
    cfg = resolve_cachejpeg_rosetta_eval_config(
        {
            "fusion_type": "latent_kv_split",
            "split_latent_cachejpeg": {
                "enabled": True,
                "codec": {
                    "compute": {"backend": "gpu"},
                    "entropy": {
                        "representation": "dense_int16",
                        "backend": "zlib1",
                    },
                },
            },
        }
    )
    assert cfg.split_latent_cachejpeg.enabled is True
    assert cfg.split_latent_cachejpeg.codec.entropy.backend == "zlib1"

    try:
        resolve_cachejpeg_rosetta_eval_config(
            {
                "fusion_type": "latent_kv_split",
                "split_latent_cachejpeg": {
                    "enabled": True,
                    "codec": {"entropy": {"backend": "lz4"}},
                },
            }
        )
    except ValueError as exc:
        assert "zlib" in str(exc)
    else:
        raise AssertionError("Expected non-zlib split latent codec to be rejected")
