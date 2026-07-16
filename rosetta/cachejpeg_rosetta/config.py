from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosetta.cachejpeg.config import CacheJPEGEvalConfig, resolve_cachejpeg_eval_config


@dataclass(frozen=True)
class CacheJPEGRosettaEvalConfig:
    sharer_model_role: str = "teacher"
    receiver_model_role: str = "base"
    homo_c2c_kv_src: str = "/data/smy/HomoC2C-KV/src"
    codec: CacheJPEGEvalConfig = field(default_factory=CacheJPEGEvalConfig)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("cachejpeg_rosetta config sections must be mappings.")
    return value


def resolve_cachejpeg_rosetta_eval_config(config: dict[str, Any]) -> CacheJPEGRosettaEvalConfig:
    nested = _mapping(config)
    codec_cfg = _mapping(nested.get("codec"))
    if "homo_c2c_kv_src" not in codec_cfg and nested.get("homo_c2c_kv_src") is not None:
        codec_cfg["homo_c2c_kv_src"] = nested["homo_c2c_kv_src"]

    return CacheJPEGRosettaEvalConfig(
        sharer_model_role=str(nested.get("sharer_model_role", "teacher")).lower(),
        receiver_model_role=str(nested.get("receiver_model_role", "base")).lower(),
        homo_c2c_kv_src=str(nested.get("homo_c2c_kv_src", "/data/smy/HomoC2C-KV/src")),
        codec=resolve_cachejpeg_eval_config(codec_cfg),
    )
