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
from .fuser_bridge import LoadedRosettaAssets, RosettaFuserBridge
from .layer_streaming import LayerCompressionPipeline, LayerPrefillTimer, StreamingDynamicCache


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
        if self.fusion_type == "latent_kv_split":
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
                num_kv_heads=int(
                    getattr(
                        self.teacher_model.config,
                        "num_key_value_heads",
                        self.teacher_model.config.num_attention_heads,
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
        self.fuser_bridge = RosettaFuserBridge(
            assets, adaptive_quant_table=adaptive_quant_table
        )
        transport_config = codec_config.get("transport") or (codec_config.get("codec") or {}).get("transport")
        self.transport = build_transport(dict(transport_config or {}))
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
    ):
        model_kwargs = {}
        if past_key_values is not None:
            model_kwargs["past_key_values"] = past_key_values
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

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **generation_config):
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
