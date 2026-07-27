from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnchorConfig:
    sink_count: int = 1
    recent_count: int = 0
    preserve_options: bool = False
    option_token_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlockConfig:
    mode: str = "global"
    size: int = 64


@dataclass(frozen=True)
class QuantConfig:
    q_global: float = 1.0
    low: float = 1.0
    high: float = 8.0
    curve: str = "quadratic"
    key_scale: float = 1.0
    value_scale: float = 1.0
    layer_group_scales: dict[str, float] = field(default_factory=dict)
    component_scale_rules: list[dict[str, Any]] = field(default_factory=list)
    clip_int16: bool = True
    clip_int8: bool = True


@dataclass(frozen=True)
class EntropyConfig:
    representation: str = "dense_int16"
    backend: str = "zlib1"


@dataclass(frozen=True)
class ComputeConfig:
    backend: str = "gpu"
    transform_dtype: str = "float32"


@dataclass(frozen=True)
class TransportConfig:
    mode: str = "none"
    timeout_seconds: float = 120.0
    max_payload_bytes: int = 8 << 30
    bandwidth_bytes_per_sec: float | None = None
    fixed_latency_ms: float = 0.0


@dataclass(frozen=True)
class CacheJPEGEvalConfig:
    method: str = "cachejpeg"
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    zero_tail: dict[str, Any] = field(default_factory=dict)
    probe: dict[str, Any] = field(default_factory=dict)
    homo_c2c_kv_src: str = "/data/smy/HomoC2C-KV/src"


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("CacheJPEG config sections must be mappings.")
    return value


def _tuple_ints(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


def resolve_cachejpeg_eval_config(config: dict[str, Any]) -> CacheJPEGEvalConfig:
    from rosetta.cachejpeg.entropy_backends import validate_entropy_backend

    anchors = _mapping(config.get("anchors"))
    block = _mapping(config.get("block"))
    quant = _mapping(config.get("quant"))
    entropy = _mapping(config.get("entropy"))
    representation = str(entropy.get("representation", "dense_int16")).lower()
    if representation not in {"dense_int16", "dense_int8", "adaptive_int"}:
        raise ValueError(
            "cachejpeg.entropy.representation must be dense_int16, dense_int8, or adaptive_int."
        )
    compute = _mapping(config.get("compute"))
    transport = _mapping(config.get("transport"))
    compute_backend = str(compute.get("backend", "gpu")).lower()
    if compute_backend not in {"gpu", "cpu"}:
        raise ValueError("cachejpeg.compute.backend must be 'gpu' or 'cpu'.")
    transform_dtype = str(compute.get("transform_dtype", "float32")).lower()
    if transform_dtype not in {"float32", "fp32"}:
        raise ValueError("GPU CacheJPEG currently requires compute.transform_dtype=float32.")

    return CacheJPEGEvalConfig(
        method=str(config.get("method", "cachejpeg")).lower(),
        anchors=AnchorConfig(
            sink_count=int(anchors.get("sink_count", config.get("sink_token_count", 1))),
            recent_count=int(anchors.get("recent_count", 0)),
            preserve_options=bool(anchors.get("preserve_options", False)),
            option_token_indices=_tuple_ints(anchors.get("option_token_indices")),
        ),
        block=BlockConfig(
            mode=str(block.get("mode", "global")).lower(),
            size=int(block.get("size", 64)),
        ),
        quant=QuantConfig(
            q_global=float(quant.get("q_global", 1.0)),
            low=float(quant.get("low", config.get("quant_table_low", 1.0))),
            high=float(quant.get("high", config.get("quant_table_high", 8.0))),
            curve=str(quant.get("curve", config.get("quant_table_curve", "quadratic"))).lower(),
            key_scale=float(quant.get("key_scale", config.get("key_quant_table_scale", 1.0))),
            value_scale=float(quant.get("value_scale", config.get("value_quant_table_scale", 1.0))),
            layer_group_scales=dict(quant.get("layer_group_scales", {})),
            component_scale_rules=list(quant.get("component_scale_rules", [])),
            clip_int16=bool(quant.get("clip_int16", True)),
            clip_int8=bool(quant.get("clip_int8", quant.get("clip_int16", True))),
        ),
        entropy=EntropyConfig(
            representation=representation,
            backend=validate_entropy_backend(str(entropy.get("backend", "zlib1"))),
        ),
        compute=ComputeConfig(
            backend=compute_backend,
            transform_dtype="float32",
        ),
        transport=TransportConfig(
            mode=str(transport.get("mode", "none")).lower(),
            timeout_seconds=float(transport.get("timeout_seconds", 120.0)),
            max_payload_bytes=int(transport.get("max_payload_bytes", 8 << 30)),
            bandwidth_bytes_per_sec=(
                float(transport["bandwidth_bytes_per_sec"])
                if transport.get("bandwidth_bytes_per_sec") is not None
                else None
            ),
            fixed_latency_ms=float(transport.get("fixed_latency_ms", 0.0)),
        ),
        zero_tail=dict(_mapping(config.get("zero_tail"))),
        probe=dict(_mapping(config.get("probe"))),
        homo_c2c_kv_src=str(config.get("homo_c2c_kv_src", "/data/smy/HomoC2C-KV/src")),
    )
