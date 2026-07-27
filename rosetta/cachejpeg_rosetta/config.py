from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rosetta.cachejpeg.config import CacheJPEGEvalConfig, resolve_cachejpeg_eval_config
from rosetta.model.latent_kv import (
    LatentKVBridgeConfig,
    resolve_latent_kv_bridge_config,
)
from rosetta.model.adaptive_quant_table import (
    AdaptiveQuantTableConfig,
    resolve_adaptive_quant_table_config,
)


@dataclass(frozen=True)
class LayerStreamingConfig:
    enabled: bool = False
    queue_size: int = 2


@dataclass(frozen=True)
class SplitLatentCacheJPEGConfig:
    enabled: bool = False
    codec: CacheJPEGEvalConfig = field(default_factory=CacheJPEGEvalConfig)


@dataclass(frozen=True)
class CacheJPEGRosettaEvalConfig:
    sharer_model_role: str = "teacher"
    receiver_model_role: str = "base"
    homo_c2c_kv_src: str = "/data/smy/HomoC2C-KV/src"
    codec: CacheJPEGEvalConfig = field(default_factory=CacheJPEGEvalConfig)
    layer_streaming: LayerStreamingConfig = field(default_factory=LayerStreamingConfig)
    fusion_type: str = "original"
    latent_kv_bridge: LatentKVBridgeConfig = field(default_factory=LatentKVBridgeConfig)
    split_latent_cachejpeg: SplitLatentCacheJPEGConfig = field(
        default_factory=SplitLatentCacheJPEGConfig
    )
    adaptive_quant_table: AdaptiveQuantTableConfig = field(
        default_factory=AdaptiveQuantTableConfig
    )


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("cachejpeg_rosetta config sections must be mappings.")
    return value


def resolve_cachejpeg_rosetta_eval_config(config: dict[str, Any]) -> CacheJPEGRosettaEvalConfig:
    nested = _mapping(config)
    codec_cfg = _mapping(nested.get("codec"))
    streaming_cfg = _mapping(nested.get("layer_streaming"))
    latent_cfg = _mapping(nested.get("latent_kv_bridge"))
    split_latent_cachejpeg_cfg = _mapping(nested.get("split_latent_cachejpeg"))
    adaptive_quant_table_cfg = _mapping(nested.get("adaptive_quant_table"))
    split_latent_codec_cfg = _mapping(split_latent_cachejpeg_cfg.get("codec"))
    split_latent_cachejpeg_enabled = bool(
        split_latent_cachejpeg_cfg.get("enabled", False)
    )
    fusion_type = str(nested.get("fusion_type", "original")).lower()
    if fusion_type not in {"original", "latent_kv_joint", "latent_kv_split"}:
        raise ValueError(
            "cachejpeg_rosetta.fusion_type must be 'original', "
            "'latent_kv_joint', or 'latent_kv_split'."
        )
    if fusion_type in {"latent_kv_joint", "latent_kv_split"}:
        if "enabled" in latent_cfg and not latent_cfg["enabled"]:
            raise ValueError(
                f"fusion_type={fusion_type!r} requires latent_kv_bridge.enabled=true."
            )
        latent_cfg["enabled"] = True
    if fusion_type == "latent_kv_split" and bool(streaming_cfg.get("enabled", False)):
        raise ValueError(
            "latent_kv_split does not support layer_streaming in this implementation."
        )
    if split_latent_cachejpeg_enabled and fusion_type != "latent_kv_split":
        raise ValueError(
            "split_latent_cachejpeg.enabled=true requires fusion_type='latent_kv_split'."
        )
    resolved_split_latent_codec = resolve_cachejpeg_eval_config(
        split_latent_codec_cfg
    )
    if (
        split_latent_cachejpeg_enabled
        and not resolved_split_latent_codec.entropy.backend.startswith("zlib")
    ):
        raise ValueError(
            "split_latent_cachejpeg requires a zlib entropy backend, for example zlib1."
        )
    if "homo_c2c_kv_src" not in codec_cfg and nested.get("homo_c2c_kv_src") is not None:
        codec_cfg["homo_c2c_kv_src"] = nested["homo_c2c_kv_src"]

    return CacheJPEGRosettaEvalConfig(
        sharer_model_role=str(nested.get("sharer_model_role", "teacher")).lower(),
        receiver_model_role=str(nested.get("receiver_model_role", "base")).lower(),
        homo_c2c_kv_src=str(nested.get("homo_c2c_kv_src", "/data/smy/HomoC2C-KV/src")),
        codec=resolve_cachejpeg_eval_config(codec_cfg),
        layer_streaming=LayerStreamingConfig(
            enabled=bool(streaming_cfg.get("enabled", False)),
            queue_size=max(1, int(streaming_cfg.get("queue_size", 2))),
        ),
        fusion_type=fusion_type,
        latent_kv_bridge=resolve_latent_kv_bridge_config(latent_cfg),
        split_latent_cachejpeg=SplitLatentCacheJPEGConfig(
            enabled=split_latent_cachejpeg_enabled,
            codec=resolved_split_latent_codec,
        ),
        adaptive_quant_table=resolve_adaptive_quant_table_config(
            adaptive_quant_table_cfg
        ),
    )
