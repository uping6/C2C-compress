from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg.wrapper import _ensure_homo_imports
from rosetta.cachejpeg.transport import build_transport
from rosetta.cachejpeg.transport import serialize_payload
from rosetta.model.latent_kv import (
    CacheAdapter,
    CacheJPEGLatentKVPayload,
    latent_payload_to_pseudo_kv_cache,
    pseudo_kv_cache_to_latent_payload,
)
from rosetta.model.projector import load_projector
from rosetta.model.adaptive_quant_table import AdaptiveCoefficientQuantizer
from rosetta.utils.evaluate import apply_generation_config, load_hf_model, set_default_chat_template

from .config import CacheJPEGRosettaEvalConfig, resolve_cachejpeg_rosetta_eval_config
from .adaptive_quant_codec import (
    AdaptiveQuantizedCachePayload,
    decode_adaptive_quantized_cache,
    encode_adaptive_quantized_cache,
)
from .cache_aligner import ConcatCacheAligner
from .projected_kv_cache_aligner import ProjectedKVConcatCacheAligner
from .direct_mlp_cache_aligner import DirectMLPConcatCacheAligner
from .fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge
from .layer_streaming import LayerCompressionPipeline, LayerPrefillTimer, StreamingDynamicCache
from .concat_layer_streaming import ConcatLayerPipeline
from .pre_rope import (
    StreamingPreRopeDynamicCache,
    StreamingPreRopeKVPublisher,
    capture_pre_rope_keys,
    replace_cache_keys_with_pre_rope,
    stream_pre_rope_keys,
)


def _hf_local_files_only() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


def _resolve_checkpoint_dir(checkpoints_dir: str, checkpoint_subfolder: Optional[str]) -> str:
    if not checkpoint_subfolder:
        return checkpoints_dir
    candidate = os.path.join(checkpoints_dir, checkpoint_subfolder)
    return candidate if os.path.isdir(candidate) else checkpoints_dir


def _load_projector_assets(checkpoint_dir: str) -> tuple[list[Any], dict[Any, Any]]:
    projector_list = []
    if os.path.isdir(checkpoint_dir):
        num_projectors = len(
            [f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)]
        )
        for proj_idx in range(num_projectors):
            json_cfg = os.path.join(checkpoint_dir, f"projector_{proj_idx}.json")
            pt_path = os.path.join(checkpoint_dir, f"projector_{proj_idx}.pt")
            proj = load_projector(json_cfg)
            state_dict = torch.load(pt_path, map_location="cpu")
            proj.load_state_dict(state_dict, strict=False)
            projector_list.append(proj)

        projector_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
        if os.path.isfile(projector_cfg_path):
            import json

            with open(projector_cfg_path, "r", encoding="utf-8") as f:
                projector_dict = json.load(f)
        else:
            projector_dict = {}
    else:
        projector_dict = {}
    return projector_list, projector_dict


def _load_rosetta_assets(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> LoadedRosettaAssets:
    rosetta_config = model_config.get("rosetta_config") or {}
    base_model_name = rosetta_config["base_model"]
    teacher_model_name = rosetta_config["teacher_model"]
    checkpoint_dir = _resolve_checkpoint_dir(
        rosetta_config["checkpoints_dir"],
        eval_config.get("rosetta_checkpoint_subfolder"),
    )

    base_model, base_tokenizer = load_hf_model(base_model_name, device=device, generation_config=generation_config)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=getattr(base_model, "dtype", None),
        local_files_only=_hf_local_files_only(),
    ).to(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        local_files_only=_hf_local_files_only(),
    )
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    set_default_chat_template(teacher_tokenizer, str(teacher_model_name))
    apply_generation_config(teacher_model, generation_config)

    projector_list, projector_dict = _load_projector_assets(checkpoint_dir)
    return LoadedRosettaAssets(
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        projector_list=projector_list,
        projector_dict=projector_dict,
        checkpoint_dir=checkpoint_dir,
    )


class CacheJPEGRosettaEvalWrapper:
    """
    Skeleton for:
      teacher/sharer prefill -> cachejpeg encode/decode -> fuser -> base/receiver generate.
    """

    def __init__(
        self,
        assets: LoadedRosettaAssets,
        codec_config: dict[str, Any],
    ):
        self.assets = assets
        self.base_model = assets.base_model
        self.base_tokenizer = assets.base_tokenizer
        self.teacher_model = assets.teacher_model
        self.teacher_tokenizer = assets.teacher_tokenizer
        self.eval_codec_config: CacheJPEGRosettaEvalConfig = resolve_cachejpeg_rosetta_eval_config(codec_config)
        self.fusion_type = self.eval_codec_config.fusion_type
        self.cache_alignment = self.eval_codec_config.cache_alignment
        concat_projector_config = dict(codec_config.get("concat_projector") or {})
        self.concat_projector_type = str(
            concat_projector_config.get("type", "lcf_first")
        ).lower()
        if self.concat_projector_type not in {
            "lcf_first",
            "lcf_projected_kv",
            "direct_pre_rope_mlp",
        }:
            raise ValueError(
                "cachejpeg_rosetta.concat_projector.type must be 'lcf_first' "
                "or 'lcf_projected_kv' or 'direct_pre_rope_mlp'."
            )
        self.direct_mlp_concat = (
            self.cache_alignment == "concat"
            and self.concat_projector_type == "direct_pre_rope_mlp"
        )
        self.split_latent_cachejpeg_enabled = (
            self.eval_codec_config.split_latent_cachejpeg.enabled
        )
        split_latent_cachejpeg_raw = dict(
            codec_config.get("split_latent_cachejpeg") or {}
        )
        self.split_latent_codec_config = dict(
            split_latent_cachejpeg_raw.get("codec") or {}
        )
        self.codec_config = {
            **codec_config.get("codec", {}),
            "homo_c2c_kv_src": self.eval_codec_config.homo_c2c_kv_src,
        }
        if self.direct_mlp_concat:
            # The direct ablation has no serialized wire representation and must
            # not import or initialize the external HomoC2C/CacheJPEG codec.
            self.codec = None
            self._to_legacy_cache = CacheAdapter.to_legacy

            def to_dynamic_cache(cache):
                if isinstance(cache, DynamicCache):
                    return cache
                return DynamicCache.from_legacy_cache(CacheAdapter.to_legacy(cache))

            self._to_dynamic_cache = to_dynamic_cache
        elif self.fusion_type == "latent_kv_split":
            # Split mode transmits LatentKVPayload directly and therefore has no
            # dependency on the CacheJPEG/HomoC2C codec implementation.
            self.codec = None
            if self.split_latent_cachejpeg_enabled:
                codec_cls, _, _, _ = _ensure_homo_imports(
                    self.eval_codec_config.homo_c2c_kv_src
                )
                if self.eval_codec_config.split_latent_cachejpeg.codec.compute.backend == "gpu":
                    from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

                    self.codec = GPUCacheJPEGCodec(
                        device=next(self.teacher_model.parameters()).device
                    )
                else:
                    self.codec = codec_cls()
            self._to_legacy_cache = CacheAdapter.to_legacy

            def to_dynamic_cache(cache):
                if isinstance(cache, DynamicCache):
                    return cache
                return DynamicCache.from_legacy_cache(CacheAdapter.to_legacy(cache))

            self._to_dynamic_cache = to_dynamic_cache
        else:
            (
                codec_cls,
                _codec_config_resolver,
                self._to_dynamic_cache,
                self._to_legacy_cache,
            ) = _ensure_homo_imports(self.eval_codec_config.homo_c2c_kv_src)
            if self.eval_codec_config.codec.compute.backend == "gpu":
                from rosetta.cachejpeg.gpu_codec import GPUCacheJPEGCodec

                self.codec = GPUCacheJPEGCodec(
                    device=next(self.teacher_model.parameters()).device
                )
            else:
                self.codec = codec_cls()
        adaptive_quant_table = None
        if self.eval_codec_config.adaptive_quant_table.enabled:
            base_config = self.base_model.config
            adaptive_quant_table = AdaptiveCoefficientQuantizer(
                num_layers=int(base_config.num_hidden_layers),
                num_kv_heads=(
                    1
                    if self.cache_alignment == "concat"
                    else int(
                        getattr(
                            self.teacher_model.config,
                            "num_key_value_heads",
                            self.teacher_model.config.num_attention_heads,
                        )
                    )
                ),
                config=self.eval_codec_config.adaptive_quant_table,
            )
            state_path = Path(
                (codec_config.get("adaptive_quant_table") or {}).get(
                    "checkpoint_path",
                    Path(assets.checkpoint_dir or "") / "adaptive_quant_table.pt",
                )
            )
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"Adaptive quantization-table checkpoint not found: {state_path}"
                )
            adaptive_quant_table.load_state_dict(
                torch.load(state_path, map_location="cpu")
            )
            adaptive_quant_table.to(next(self.base_model.parameters()).device).eval()
        self.adaptive_quant_table = adaptive_quant_table
        self.fuser_bridge = (
            None
            if self.direct_mlp_concat
            else RosettaFuserBridge(
                assets, adaptive_quant_table=adaptive_quant_table
            )
        )
        self.concat_cache_aligner = None
        if self.cache_alignment == "concat":
            if self.concat_projector_type == "direct_pre_rope_mlp":
                aligner_class = DirectMLPConcatCacheAligner
            elif self.concat_projector_type == "lcf_projected_kv":
                aligner_class = ProjectedKVConcatCacheAligner
            else:
                aligner_class = ConcatCacheAligner
            self.concat_cache_aligner = aligner_class(assets)
        if self.cache_alignment == "concat" and not assets.projector_list:
            raise ValueError(
                "cache_alignment='concat' requires LCF-first projector checkpoints."
            )
        if self.cache_alignment == "concat":
            expected_concat_projector = {
                "lcf_first": "LCFFirstProjector",
                "lcf_projected_kv": "LCFProjectedKVProjector",
                "direct_pre_rope_mlp": "DirectPreRopeMLPProjector",
            }[self.concat_projector_type]
            unexpected = sorted(
                {
                    projector.__class__.__name__
                    for projector in assets.projector_list
                    if projector.__class__.__name__ != expected_concat_projector
                }
            )
            if unexpected:
                raise ValueError(
                    f"concat_projector.type={self.concat_projector_type!r} requires "
                    f"{expected_concat_projector}, "
                    f"but loaded {unexpected}."
                )
        transport_config = codec_config.get("transport") or (codec_config.get("codec") or {}).get("transport")
        self.transport = (
            None
            if self.direct_mlp_concat
            else build_transport(dict(transport_config or {}))
        )
        self.last_transport_stats = None
        self.last_codec_stats: dict[str, Any] | None = None
        self.last_fusion_stats: dict[str, Any] | None = None
        self.ablation_config = dict(codec_config.get("ablation") or {})
        if self.fusion_type in {"latent_kv_joint", "latent_kv_split"}:
            if not assets.projector_list:
                raise ValueError(
                    f"fusion_type={self.fusion_type!r} requires a compatible "
                    "latent KV checkpoint."
                )
            expected_class = (
                "LatentKVCompressor"
                if self.fusion_type == "latent_kv_joint"
                else "SplitLatentKVProjector"
            )
            unexpected = [
                projector.__class__.__name__
                for projector in assets.projector_list
                if projector.__class__.__name__ != expected_class
            ]
            if unexpected:
                raise ValueError(
                    f"fusion_type={self.fusion_type!r} requires {expected_class}, "
                    f"but loaded {sorted(set(unexpected))}."
                )
        if self.eval_codec_config.layer_streaming.enabled and not hasattr(self.codec, "encode_layer"):
            raise ValueError(
                "cachejpeg_rosetta.layer_streaming currently requires compute.backend=gpu."
            )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @staticmethod
    def _split_aligned_inputs(input_ids, attention_mask=None):
        """Split TokenAligner output into receiver and sharer inputs."""
        if isinstance(input_ids, (list, tuple)):
            if len(input_ids) != 2:
                raise ValueError(
                    "CacheJPEG-Rosetta aligned input_ids must contain base and teacher tensors"
                )
            base_input_ids, teacher_input_ids = input_ids
            if isinstance(attention_mask, (list, tuple)):
                if len(attention_mask) != 2:
                    raise ValueError(
                        "CacheJPEG-Rosetta aligned attention_mask must contain base and teacher tensors"
                    )
                base_attention_mask, teacher_attention_mask = attention_mask
            else:
                base_attention_mask = teacher_attention_mask = attention_mask
            return base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask
        return input_ids, input_ids, attention_mask, attention_mask

    def _configured_frequency_prune_stats(self) -> dict[str, Any]:
        eval_cfg = getattr(self, "eval_codec_config", None)
        codec_eval_cfg = getattr(eval_cfg, "codec", None)
        prune_eval_cfg = getattr(codec_eval_cfg, "frequency_prune", None)
        if prune_eval_cfg is not None:
            return {
                "enabled": bool(prune_eval_cfg.enabled),
                "prune_from": str(prune_eval_cfg.prune_from),
            }
        raw_prune = (getattr(self, "codec_config", {}) or {}).get(
            "frequency_prune", {}
        )
        return {
            "enabled": bool(raw_prune.get("enabled", False)),
            "prune_from": str(
                raw_prune.get("prune_from", raw_prune.get("prune_from_band", "B4"))
            ).upper(),
        }

    def _decode_from_receiver_cache(
        self,
        last_token: torch.Tensor,
        past_key_values,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        generated = []
        current_input = last_token
        current_past = self._to_dynamic_cache(past_key_values)
        for _ in range(max(1, int(max_new_tokens))):
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=current_input,
                    past_key_values=current_past,
                    use_cache=True,
                )
            logits = outputs.logits[:, -1, :]
            if do_sample:
                scaled = logits if temperature <= 0 else logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token)
            current_input = next_token
            current_past = self._to_dynamic_cache(outputs.past_key_values)
            if self.base_tokenizer.eos_token_id is not None and next_token.item() == self.base_tokenizer.eos_token_id:
                break
        if not generated:
            return last_token
        return torch.cat(generated, dim=1)

    def prefill_on_sharer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        *,
        pre_rope: bool = False,
    ):
        model_kwargs = {}
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
        if pre_rope:
            if past_key_values is not None:
                raise ValueError("pre-RoPE sharer capture does not support an existing cache.")
            with capture_pre_rope_keys(self.teacher_model) as captured:
                with torch.no_grad():
                    outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
            outputs.past_key_values = replace_cache_keys_with_pre_rope(
                outputs.past_key_values, captured
            )
            return outputs
        with torch.no_grad():
            return self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                **model_kwargs,
            )

    def prefill_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

    def encode_cache(self, past_key_values):
        legacy_cache = self._to_legacy_cache(past_key_values)
        if not getattr(self.codec, "uses_gpu_transform", False):
            legacy_cache = self._cache_to_codec_dtype(legacy_cache)
        return self.codec.encode(legacy_cache, self.codec_config)

    def decode_cache(self, payload):
        return self.codec.decode(payload, self.codec_config)

    @staticmethod
    def _cache_to_codec_dtype(past_key_values):
        return tuple(
            (
                key.detach().to(dtype=torch.float32),
                value.detach().to(dtype=torch.float32),
            )
            for key, value in past_key_values
        )

    def fuse_to_receiver_cache(self, decoded_teacher_cache, base_seed_cache=None):
        fused = self.fuser_bridge.fuse_teacher_cache_to_base(
            decoded_teacher_cache,
            base_seed_cache=base_seed_cache,
        )
        self.last_fusion_stats = getattr(
            self.fuser_bridge, "last_fusion_stats", None
        )
        return fused

    def generate_on_receiver(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )

    def _generate_with_layer_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_aligned_inputs(input_ids, attention_mask)
        )
        num_layers = int(self.teacher_model.config.num_hidden_layers)
        pipeline = LayerCompressionPipeline(
            codec=self.codec,
            codec_config=self.codec_config,
            transport=self.transport,
            num_layers=num_layers,
            queue_size=self.eval_codec_config.layer_streaming.queue_size,
        )
        streaming_cache = StreamingDynamicCache(pipeline.submit)
        prefill_timer = LayerPrefillTimer(self.teacher_model, num_layers)
        prefill_timer.start()
        pipeline_started = time.perf_counter()
        try:
            self.prefill_on_sharer(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                past_key_values=streaming_cache,
            )
            # Receiver prefill overlaps with any compression work still queued.
            receiver_seed_outputs = self.prefill_on_receiver(
                input_ids=base_input_ids, attention_mask=base_attention_mask
            )
            decoded_teacher_cache = pipeline.finish()
            teacher_device = next(self.teacher_model.parameters()).device
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            layer_prefill_seconds = prefill_timer.finish()
        except BaseException:
            for handle in prefill_timer.handles:
                handle.remove()
            prefill_timer.handles.clear()
            pipeline.abort()
            raise
        pipeline_seconds = time.perf_counter() - pipeline_started
        self.last_transport_stats = pipeline.aggregate_transport_stats()

        zero_sharer_cache = bool(self.ablation_config.get("zero_sharer_cache_at_receiver", False))
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in decoded_teacher_cache
            )
        self.last_codec_stats = {
            "mode": "layer_streaming",
            "num_layers": num_layers,
            "queue_size": self.eval_codec_config.layer_streaming.queue_size,
            "frequency_prune": self._configured_frequency_prune_stats(),
            "original_kv_bytes": pipeline.original_kv_bytes,
            "payload_bytes": pipeline.payload_bytes,
            "compression_factor": (
                float(pipeline.original_kv_bytes / pipeline.payload_bytes)
                if pipeline.payload_bytes
                else 0.0
            ),
            "space_saving_ratio": (
                float(1.0 - pipeline.payload_bytes / pipeline.original_kv_bytes)
                if pipeline.original_kv_bytes
                else 0.0
            ),
            "encode_seconds": float(pipeline.encode_seconds),
            "avg_layer_encode_seconds": float(
                sum(value for value in pipeline.layer_encode_seconds if value is not None)
                / max(1, sum(value is not None for value in pipeline.layer_encode_seconds))
            ),
            "layer_encode_seconds": [
                float(value) if value is not None else None
                for value in pipeline.layer_encode_seconds
            ],
            "avg_layer_prefill_seconds": float(sum(layer_prefill_seconds) / len(layer_prefill_seconds)),
            "layer_prefill_seconds": [float(value) for value in layer_prefill_seconds],
            "decode_seconds": float(pipeline.decode_seconds),
            "pipeline_seconds": float(pipeline_seconds),
            "transport_bandwidth_bytes_per_sec": getattr(
                self.transport, "bandwidth_bytes_per_sec", None
            ),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)

    def _generate_with_split_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Transmit only sharer-produced latent tensors, then fuse at receiver."""

        # Keep compatibility with lightweight wrappers created via ``__new__``
        # in callers that predate the optional CacheJPEG path.
        split_latent_cachejpeg_enabled = bool(
            getattr(self, "split_latent_cachejpeg_enabled", False)
        )

        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_aligned_inputs(input_ids, attention_mask)
        )
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
        )
        receiver_seed_outputs = self.prefill_on_receiver(
            input_ids=base_input_ids, attention_mask=base_attention_mask
        )
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )

        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_started = time.perf_counter()
        latent_payload = self.fuser_bridge.encode_teacher_cache_to_latents(
            sharer_outputs.past_key_values,
            move_to_cpu=not split_latent_cachejpeg_enabled,
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started

        latent_bytes = sum(
            int(layer.latent.numel() * layer.latent.element_size())
            for layer in latent_payload.layers
        )
        cachejpeg_encode_seconds = 0.0
        wire_payload = latent_payload
        cachejpeg_summary = None
        if split_latent_cachejpeg_enabled:
            pseudo_cache = latent_payload_to_pseudo_kv_cache(latent_payload)
            if not getattr(self.codec, "uses_gpu_transform", False):
                pseudo_cache = self._cache_to_codec_dtype(pseudo_cache)
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            cachejpeg_encode_started = time.perf_counter()
            encoded_payload = self.codec.encode(
                pseudo_cache, self.split_latent_codec_config
            )
            if teacher_device.type == "cuda":
                torch.cuda.synchronize(teacher_device)
            cachejpeg_encode_seconds = (
                time.perf_counter() - cachejpeg_encode_started
            )
            cachejpeg_summary = dict(
                getattr(encoded_payload, "local_summary", {}) or {}
            )
            wire_payload = CacheJPEGLatentKVPayload(
                encoded_payload=encoded_payload,
                layers=[
                    (
                        int(layer.receiver_layer),
                        int(layer.sharer_layer),
                        int(layer.projector_idx),
                    )
                    for layer in latent_payload.layers
                ],
                latent_dim=latent_payload.latent_dim,
                sequence_length=latent_payload.sequence_length,
                source_dtype=latent_payload.source_dtype,
                entropy_backend=(
                    self.eval_codec_config.split_latent_cachejpeg.codec.entropy.backend
                ),
            )

        payload_bytes = len(serialize_payload(wire_payload))
        transport = getattr(self, "transport", None)
        received_payload = (
            transport.roundtrip(wire_payload)
            if transport is not None
            else wire_payload
        )
        self.last_transport_stats = (
            transport.last_stats if transport is not None else None
        )

        cachejpeg_decode_seconds = 0.0
        decoder_payload = received_payload
        receiver_device = next(self.base_model.parameters()).device
        if split_latent_cachejpeg_enabled:
            if not isinstance(received_payload, CacheJPEGLatentKVPayload):
                raise TypeError(
                    "Expected CacheJPEGLatentKVPayload after split transport, got "
                    f"{type(received_payload)!r}."
                )
            if receiver_device.type == "cuda":
                torch.cuda.synchronize(receiver_device)
            cachejpeg_decode_started = time.perf_counter()
            reconstructed_pseudo_cache = self.codec.decode(
                received_payload.encoded_payload,
                self.split_latent_codec_config,
            )
            if receiver_device.type == "cuda":
                torch.cuda.synchronize(receiver_device)
            cachejpeg_decode_seconds = (
                time.perf_counter() - cachejpeg_decode_started
            )
            decoder_payload = pseudo_kv_cache_to_latent_payload(
                reconstructed_pseudo_cache, received_payload
            )

        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        decode_started = time.perf_counter()
        fused_receiver_cache = self.fuser_bridge.fuse_latents_to_base(
            decoder_payload,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        decode_seconds = time.perf_counter() - decode_started
        self.last_fusion_stats = self.fuser_bridge.last_fusion_stats
        self.last_codec_stats = {
            "mode": (
                "latent_kv_split_cachejpeg"
                if split_latent_cachejpeg_enabled
                else "latent_kv_split"
            ),
            "quantized": split_latent_cachejpeg_enabled,
            "entropy_backend": (
                self.eval_codec_config.split_latent_cachejpeg.codec.entropy.backend
                if split_latent_cachejpeg_enabled
                else None
            ),
            "original_kv_bytes": original_kv_bytes,
            "latent_bytes": latent_bytes,
            "metadata_bytes": (
                None
                if split_latent_cachejpeg_enabled
                else max(0, payload_bytes - latent_bytes)
            ),
            "payload_bytes": payload_bytes,
            "compression_factor": (
                float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0
            ),
            "latent_compression_factor": (
                float(latent_bytes / payload_bytes) if payload_bytes else 0.0
            ),
            "latent_element_compression_factor": (
                float(
                    sum(key.numel() + value.numel() for key, value in legacy_sharer_cache)
                    / sum(layer.latent.numel() for layer in latent_payload.layers)
                )
                if latent_payload.layers
                else 0.0
            ),
            "encode_seconds": float(
                encode_seconds + cachejpeg_encode_seconds
            ),
            "decode_seconds": float(
                cachejpeg_decode_seconds + decode_seconds
            ),
            "latent_encode_seconds": float(encode_seconds),
            "cachejpeg_encode_seconds": float(cachejpeg_encode_seconds),
            "cachejpeg_decode_seconds": float(cachejpeg_decode_seconds),
            "receiver_decode_seconds": float(decode_seconds),
            "cachejpeg_summary": cachejpeg_summary,
        }
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)

    def _decode_from_prefill_outputs(
        self,
        prefill_outputs,
        *,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
    ) -> torch.Tensor:
        """Decode without feeding the receiver prompt's final token a second time."""

        generated = []
        current_past = self._to_dynamic_cache(prefill_outputs.past_key_values)
        current_logits = prefill_outputs.logits[:, -1, :]
        current_attention_mask = attention_mask
        eos_token_id = self.base_tokenizer.eos_token_id
        for generated_index in range(max(1, int(max_new_tokens))):
            if do_sample:
                scaled = current_logits if temperature <= 0 else current_logits / temperature
                next_token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(current_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_token_id is not None and bool(torch.all(next_token == int(eos_token_id))):
                break
            if generated_index + 1 == max(1, int(max_new_tokens)):
                break
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    past_key_values=current_past,
                    use_cache=True,
                )
            current_past = self._to_dynamic_cache(outputs.past_key_values)
            current_logits = outputs.logits[:, -1, :]
        return torch.cat(generated, dim=1)

    def _generate_with_direct_mlp_concat(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Project Sharer pre-RoPE KV directly into an uncompressed prefix."""

        base_ids, teacher_ids, base_mask, teacher_mask = self._split_aligned_inputs(
            input_ids, attention_mask
        )
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pre_rope=True,
        )
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )
        if not isinstance(self.concat_cache_aligner, DirectMLPConcatCacheAligner):
            raise RuntimeError("Direct MLP concat aligner was not initialized.")

        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        projection_started = time.perf_counter()
        receiver_prefix = self.concat_cache_aligner.align(legacy_sharer_cache)
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        projection_seconds = time.perf_counter() - projection_started
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(receiver_prefix.key_cache[0].shape[2])

        if base_mask is None:
            base_mask = torch.ones_like(base_ids, dtype=torch.long)
        if teacher_mask is not None and teacher_mask.shape[1] == prefix_length:
            prefix_mask = teacher_mask.to(
                device=base_mask.device, dtype=base_mask.dtype
            )
        else:
            prefix_mask = torch.ones(
                (base_ids.shape[0], prefix_length),
                dtype=base_mask.dtype,
                device=base_mask.device,
            )
        combined_mask = torch.cat((prefix_mask, base_mask), dim=1)
        receiver_position_ids = (
            base_mask.long().cumsum(-1) - 1 + prefix_length
        ).masked_fill(base_mask == 0, 0)
        cache_position = torch.arange(
            prefix_length,
            prefix_length + base_ids.shape[1],
            device=base_ids.device,
        )
        with torch.no_grad():
            receiver_outputs = self.base_model(
                input_ids=base_ids,
                attention_mask=combined_mask,
                position_ids=receiver_position_ids,
                cache_position=cache_position,
                past_key_values=receiver_prefix,
                use_cache=True,
            )
        generated = self._decode_from_prefill_outputs(
            receiver_outputs,
            attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        self.last_transport_stats = None
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "lcf_latent_kv_bytes": None,
            "payload_bytes": None,
            "compression_factor": None,
            "space_saving_ratio": None,
            "encode_seconds": None,
            "decode_seconds": None,
            "mlp_projection_seconds": float(projection_seconds),
            "cache_alignment": "concat",
            "concat_projector_type": "direct_pre_rope_mlp",
            "communication_mode": "local_direct",
            "codec_order": "direct_mlp",
            "rope_mode": "pre_rope",
            "prefix_tokens": prefix_length,
            "layer_streaming": False,
            "transport_mode": None,
        }
        return torch.cat([base_ids, generated], dim=1)

    def _generate_with_concat_alignment(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Use mapped Sharer pre-RoPE KV as a prefix to Receiver prefill and decode."""

        base_ids, teacher_ids, base_mask, teacher_mask = self._split_aligned_inputs(
            input_ids, attention_mask
        )
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_ids,
            attention_mask=teacher_mask,
            pre_rope=True,
        )
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        if self.concat_cache_aligner is None:
            raise RuntimeError("Concat cache aligner was not initialized.")
        lcf_encode_started = time.perf_counter()
        lcf_latent_cache, lcf_routing = self.concat_cache_aligner.encode(
            legacy_sharer_cache
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        lcf_encode_seconds = time.perf_counter() - lcf_encode_started
        latent_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in lcf_latent_cache
        )
        encode_started = time.perf_counter()
        adaptive_result = None
        if self.adaptive_quant_table is not None:
            payload, adaptive_result = encode_adaptive_quantized_cache(
                self.adaptive_quant_table,
                lcf_latent_cache,
                representation=self.eval_codec_config.codec.entropy.representation,
                backend=self.eval_codec_config.codec.entropy.backend,
            )
        else:
            payload = self.encode_cache(lcf_latent_cache)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started
        payload_bytes = len(serialize_payload(payload))
        received_payload = self.transport.roundtrip(payload) if self.transport is not None else payload
        self.last_transport_stats = (
            self.transport.last_stats if self.transport is not None else None
        )
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_started = time.perf_counter()
        if self.adaptive_quant_table is not None:
            if not isinstance(received_payload, AdaptiveQuantizedCachePayload):
                raise TypeError(
                    "Expected AdaptiveQuantizedCachePayload after transport, got "
                    f"{type(received_payload)!r}."
                )
            decoded_latent_cache = decode_adaptive_quantized_cache(
                received_payload,
                self.adaptive_quant_table,
                device=teacher_device,
                dtype=lcf_latent_cache[0][0].dtype,
            )
        else:
            decoded_latent_cache = self.decode_cache(received_payload)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_seconds = time.perf_counter() - decode_started

        if bool(self.ablation_config.get("zero_sharer_cache_at_receiver", False)):
            decoded_latent_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in self._to_legacy_cache(decoded_latent_cache)
            )
        lcf_decode_started = time.perf_counter()
        receiver_prefix = self.concat_cache_aligner.decode(
            decoded_latent_cache, lcf_routing
        )
        receiver_device = next(self.base_model.parameters()).device
        if receiver_device.type == "cuda":
            torch.cuda.synchronize(receiver_device)
        lcf_decode_seconds = time.perf_counter() - lcf_decode_started
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(receiver_prefix.key_cache[0].shape[2])

        if base_mask is None:
            base_mask = torch.ones_like(base_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (base_ids.shape[0], prefix_length),
            dtype=base_mask.dtype,
            device=base_mask.device,
        )
        combined_mask = torch.cat([prefix_mask, base_mask], dim=1)
        receiver_position_ids = base_mask.long().cumsum(-1) - 1 + prefix_length
        receiver_position_ids = receiver_position_ids.masked_fill(base_mask == 0, 0)
        with torch.no_grad():
            receiver_outputs = self.base_model(
                input_ids=base_ids,
                attention_mask=combined_mask,
                position_ids=receiver_position_ids,
                past_key_values=receiver_prefix,
                use_cache=True,
            )
        generated = self._decode_from_prefill_outputs(
            receiver_outputs,
            attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "lcf_latent_kv_bytes": latent_kv_bytes,
            "payload_bytes": payload_bytes,
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "space_saving_ratio": float(1.0 - payload_bytes / original_kv_bytes) if original_kv_bytes else 0.0,
            "encode_seconds": float(encode_seconds),
            "decode_seconds": float(decode_seconds),
            "lcf_encode_seconds": float(lcf_encode_seconds),
            "lcf_decode_seconds": float(lcf_decode_seconds),
            "sender_encode_seconds": float(lcf_encode_seconds + encode_seconds),
            "receiver_decode_seconds": float(decode_seconds + lcf_decode_seconds),
            "compute_backend": self.eval_codec_config.codec.compute.backend,
            "transform_dtype": self.eval_codec_config.codec.compute.transform_dtype,
            "entropy_backend": self.eval_codec_config.codec.entropy.backend,
            "layer_streaming": False,
            "layer_execution": "whole_cache_sequential",
            "transport_mode": self.eval_codec_config.codec.transport.mode,
            "transport_bandwidth_bytes_per_sec": (
                self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
            ),
            "bandwidth_only_transmit_seconds": (
                float(
                    payload_bytes
                    / self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
                )
                if self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
                else None
            ),
            "cache_alignment": "concat",
            "concat_projector_type": self.concat_projector_type,
            "adaptive_quant_table_enabled": self.adaptive_quant_table is not None,
            "adaptive_quant_estimated_payload_bits": (
                float(adaptive_result.estimated_payload_bits.item())
                if adaptive_result is not None
                else None
            ),
            "adaptive_quant_mean_alpha": (
                float(adaptive_result.alpha.float().mean().item())
                if adaptive_result is not None
                else None
            ),
            "adaptive_quant_fixed_alpha": (
                self.adaptive_quant_table.config.fixed_alpha
                if self.adaptive_quant_table is not None
                else None
            ),
            "codec_order": (
                "lcf_project_kv_adaptive_quant_lcf_up"
                if self.adaptive_quant_table is not None
                else (
                    "lcf_project_kv_cachejpeg_lcf_up"
                    if self.concat_projector_type == "lcf_projected_kv"
                    else "lcf_down_cachejpeg_lcf_up"
                )
            ),
            "rope_mode": "pre_rope",
            "prefix_tokens": prefix_length,
            "latent_dim": lcf_routing.latent_dim,
        }
        return torch.cat([base_ids, generated], dim=1)

    def _generate_with_concat_layer_streaming(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: dict[str, Any],
    ) -> torch.Tensor:
        """Overlap per-layer pre-RoPE concat compression with Sharer prefill."""

        base_ids, teacher_ids, base_mask, teacher_mask = self._split_aligned_inputs(
            input_ids, attention_mask
        )
        if self.concat_cache_aligner is None:
            raise RuntimeError("Concat cache aligner was not initialized.")
        teacher_parameter = next(self.teacher_model.parameters())
        routing = self.concat_cache_aligner.prepare_routing()
        self.concat_cache_aligner.prepare_projectors(
            teacher_parameter.device, teacher_parameter.dtype
        )
        zero_sharer_cache = bool(
            self.ablation_config.get("zero_sharer_cache_at_receiver", False)
        )
        streaming = self.eval_codec_config.layer_streaming
        pipeline = ConcatLayerPipeline(
            aligner=self.concat_cache_aligner,
            codec=self.codec,
            codec_config=self.codec_config,
            transport=self.transport,
            routing=routing,
            gpu_streams=streaming.gpu_streams,
            max_inflight_layers=streaming.max_inflight_layers,
            zero_sharer_cache_at_receiver=zero_sharer_cache,
        )
        publisher = StreamingPreRopeKVPublisher(pipeline.submit)
        streaming_cache = StreamingPreRopeDynamicCache(publisher)
        prefill_timer = LayerPrefillTimer(
            self.teacher_model, int(self.teacher_model.config.num_hidden_layers)
        )
        prefill_timer.start()
        finish_attempted = False
        try:
            with stream_pre_rope_keys(self.teacher_model, publisher):
                with torch.no_grad():
                    self.teacher_model(
                        input_ids=teacher_ids,
                        attention_mask=teacher_mask,
                        past_key_values=streaming_cache,
                        use_cache=True,
                    )
            if teacher_parameter.device.type == "cuda":
                torch.cuda.synchronize(teacher_parameter.device)
            layer_prefill_seconds = prefill_timer.finish()
            finish_attempted = True
            receiver_prefix, routing = pipeline.finish()
        except BaseException:
            for handle in prefill_timer.handles:
                handle.remove()
            prefill_timer.handles.clear()
            if not finish_attempted:
                try:
                    pipeline.finish()
                except BaseException:
                    pass
            raise
        pipeline_seconds = pipeline.pipeline_seconds
        self.last_transport_stats = pipeline.aggregate_transport_stats()
        self.last_fusion_stats = self.concat_cache_aligner.last_alignment_stats
        prefix_length = int(receiver_prefix.key_cache[0].shape[2])

        if base_mask is None:
            base_mask = torch.ones_like(base_ids, dtype=torch.long)
        prefix_mask = torch.ones(
            (base_ids.shape[0], prefix_length),
            dtype=base_mask.dtype,
            device=base_mask.device,
        )
        combined_mask = torch.cat([prefix_mask, base_mask], dim=1)
        receiver_position_ids = base_mask.long().cumsum(-1) - 1 + prefix_length
        receiver_position_ids = receiver_position_ids.masked_fill(base_mask == 0, 0)
        with torch.no_grad():
            receiver_outputs = self.base_model(
                input_ids=base_ids,
                attention_mask=combined_mask,
                position_ids=receiver_position_ids,
                past_key_values=receiver_prefix,
                use_cache=True,
            )
        generated = self._decode_from_prefill_outputs(
            receiver_outputs,
            attention_mask=combined_mask,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )

        bandwidth = self.eval_codec_config.codec.transport.bandwidth_bytes_per_sec
        ordered_timings = [
            {"layer_idx": layer_idx, **pipeline.layer_timings.get(layer_idx, {})}
            for layer_idx in range(pipeline.num_layers)
        ]
        codec_encode_wall = pipeline.stage_wall_seconds("encode")
        codec_decode_wall = pipeline.stage_wall_seconds("decode")
        lcf_encode_wall = pipeline.stage_wall_seconds("lcf_encode")
        lcf_decode_wall = pipeline.stage_wall_seconds("lcf_decode")
        sender_encode_wall = pipeline.stage_wall_seconds("lcf_encode", "encode")
        receiver_decode_wall = pipeline.stage_wall_seconds("decode", "lcf_decode")
        self.last_codec_stats = {
            "mode": "concat_layer_streaming",
            "num_layers": pipeline.num_layers,
            "gpu_streams": streaming.gpu_streams,
            "max_inflight_layers": streaming.max_inflight_layers,
            "original_kv_bytes": pipeline.original_kv_bytes,
            "lcf_latent_kv_bytes": pipeline.latent_kv_bytes,
            "payload_bytes": pipeline.payload_bytes,
            "compression_factor": (
                float(pipeline.original_kv_bytes / pipeline.payload_bytes)
                if pipeline.payload_bytes
                else 0.0
            ),
            "space_saving_ratio": (
                float(1.0 - pipeline.payload_bytes / pipeline.original_kv_bytes)
                if pipeline.original_kv_bytes
                else 0.0
            ),
            "encode_seconds": codec_encode_wall,
            "decode_seconds": codec_decode_wall,
            "lcf_encode_seconds": lcf_encode_wall,
            "lcf_decode_seconds": lcf_decode_wall,
            "sender_encode_seconds": sender_encode_wall,
            "receiver_decode_seconds": receiver_decode_wall,
            "encode_service_seconds": float(pipeline.codec_encode_seconds),
            "decode_service_seconds": float(pipeline.codec_decode_seconds),
            "lcf_encode_service_seconds": float(pipeline.lcf_encode_seconds),
            "lcf_decode_service_seconds": float(pipeline.lcf_decode_seconds),
            "pipeline_seconds": float(pipeline_seconds),
            "layer_prefill_seconds": [float(value) for value in layer_prefill_seconds],
            "layer_timings": ordered_timings,
            "compute_backend": self.eval_codec_config.codec.compute.backend,
            "transform_dtype": self.eval_codec_config.codec.compute.transform_dtype,
            "entropy_backend": self.eval_codec_config.codec.entropy.backend,
            "layer_streaming": True,
            "layer_execution": "concat_staged_pipeline",
            "transport_mode": self.eval_codec_config.codec.transport.mode,
            "transport_bandwidth_bytes_per_sec": bandwidth,
            "bandwidth_only_transmit_seconds": (
                float(pipeline.payload_bytes / bandwidth) if bandwidth else None
            ),
            "cache_alignment": "concat",
            "concat_projector_type": self.concat_projector_type,
            "codec_order": (
                "lcf_project_kv_cachejpeg_lcf_up"
                if self.concat_projector_type == "lcf_projected_kv"
                else "lcf_down_cachejpeg_lcf_up"
            ),
            "rope_mode": "pre_rope",
            "prefix_tokens": prefix_length,
            "latent_dim": routing.latent_dim,
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
        }
        return torch.cat([base_ids, generated], dim=1)

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
        if getattr(self, "cache_alignment", "fuser") == "concat":
            if getattr(self, "concat_projector_type", "lcf_first") == "direct_pre_rope_mlp":
                return self._generate_with_direct_mlp_concat(
                    input_ids, attention_mask, generation_config
                )
            streaming_config = getattr(
                getattr(self, "eval_codec_config", None), "layer_streaming", None
            )
            if streaming_config is not None and streaming_config.enabled:
                return self._generate_with_concat_layer_streaming(
                    input_ids, attention_mask, generation_config
                )
            return self._generate_with_concat_alignment(
                input_ids, attention_mask, generation_config
            )
        if getattr(self, "fusion_type", "original") == "latent_kv_split":
            return self._generate_with_split_latent(
                input_ids, attention_mask, generation_config
            )
        streaming_config = getattr(getattr(self, "eval_codec_config", None), "layer_streaming", None)
        if streaming_config is not None and streaming_config.enabled:
            return self._generate_with_layer_streaming(
                input_ids, attention_mask, generation_config
            )
        base_input_ids, teacher_input_ids, base_attention_mask, teacher_attention_mask = (
            self._split_aligned_inputs(input_ids, attention_mask)
        )
        sharer_outputs = self.prefill_on_sharer(
            input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
        )
        receiver_seed_outputs = self.prefill_on_receiver(
            input_ids=base_input_ids, attention_mask=base_attention_mask
        )
        legacy_sharer_cache = self._to_legacy_cache(sharer_outputs.past_key_values)
        original_kv_bytes = sum(
            int(key.numel() * key.element_size() + value.numel() * value.element_size())
            for key, value in legacy_sharer_cache
        )
        teacher_device = next(self.teacher_model.parameters()).device
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_started = time.perf_counter()
        payload = self.encode_cache(legacy_sharer_cache)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        encode_seconds = time.perf_counter() - encode_started
        payload_bytes = len(serialize_payload(payload))
        transport = getattr(self, "transport", None)
        received_payload = transport.roundtrip(payload) if transport is not None else payload
        self.last_transport_stats = transport.last_stats if transport is not None else None
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_started = time.perf_counter()
        decoded_teacher_cache = self.decode_cache(received_payload)
        if teacher_device.type == "cuda":
            torch.cuda.synchronize(teacher_device)
        decode_seconds = time.perf_counter() - decode_started
        zero_sharer_cache = bool(
            getattr(self, "ablation_config", {}).get("zero_sharer_cache_at_receiver", False)
        )
        if zero_sharer_cache:
            decoded_teacher_cache = tuple(
                (torch.zeros_like(key), torch.zeros_like(value))
                for key, value in self._to_legacy_cache(decoded_teacher_cache)
            )
        frequency_prune_stats = getattr(payload, "local_summary", {}).get(
            "frequency_prune"
        )
        if frequency_prune_stats is None:
            frequency_prune_stats = self._configured_frequency_prune_stats()
        self.last_codec_stats = {
            "original_kv_bytes": original_kv_bytes,
            "payload_bytes": payload_bytes,
            "compression_factor": float(original_kv_bytes / payload_bytes) if payload_bytes else 0.0,
            "space_saving_ratio": float(1.0 - payload_bytes / original_kv_bytes) if original_kv_bytes else 0.0,
            "encode_seconds": float(encode_seconds),
            "decode_seconds": float(decode_seconds),
            "zero_sharer_cache_at_receiver": zero_sharer_cache,
            "frequency_prune": dict(frequency_prune_stats),
        }
        fused_receiver_cache = self.fuse_to_receiver_cache(
            decoded_teacher_cache,
            base_seed_cache=receiver_seed_outputs.past_key_values,
        )
        generated = self._decode_from_receiver_cache(
            last_token=base_input_ids[:, -1:],
            past_key_values=fused_receiver_cache,
            max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
            do_sample=bool(generation_config.get("do_sample", False)),
            temperature=float(generation_config.get("temperature", 0.0)),
        )
        return torch.cat([base_input_ids, generated], dim=1)


def load_cachejpeg_rosetta_model(
    model_config: Dict[str, Any],
    eval_config: Dict[str, Any],
    device: torch.device,
    generation_config: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    cfg = dict(model_config.get("cachejpeg_rosetta_config") or {})
    ablation_config = dict(cfg.get("ablation") or {})
    receiver_only = bool(ablation_config.get("receiver_only", False))
    sharer_only = bool(ablation_config.get("sharer_only", False))
    if receiver_only and sharer_only:
        raise ValueError("receiver_only and sharer_only cannot both be enabled")

    rosetta_config = dict(model_config.get("rosetta_config") or {})
    if receiver_only:
        base_model_name = rosetta_config["base_model"]
        base_model, base_tokenizer = load_hf_model(
            base_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the sharer/teacher,
        # projector, fuser, CacheJPEG codec, or transport.
        return base_model, base_tokenizer
    if sharer_only:
        teacher_model_name = rosetta_config["teacher_model"]
        teacher_model, teacher_tokenizer = load_hf_model(
            teacher_model_name,
            device=device,
            generation_config=generation_config,
        )
        # This branch intentionally does not load or execute the receiver/base,
        # projector, fuser, CacheJPEG codec, or transport. Returning the teacher
        # tokenizer also ensures the LongBench prompt is tokenized in the
        # sharer's own vocabulary.
        return teacher_model, teacher_tokenizer
    assets = _load_rosetta_assets(model_config, eval_config, device=device, generation_config=generation_config)
    wrapper = CacheJPEGRosettaEvalWrapper(
        assets=assets,
        codec_config=cfg,
    )
    return wrapper, assets.base_tokenizer
