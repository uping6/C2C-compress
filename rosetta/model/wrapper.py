"""
The ensemble of multiple standard transformers LLM models, with automatic kv-cache projection. It shares the same interface as the standard transformers LLM models.
"""

from typing import List, Optional, Union
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json

from rosetta.model.projector import Projector
from rosetta.model.adaptive_quant_table import AdaptiveCoefficientQuantizer
from rosetta.model.sampling import sample_token
from rosetta.model.transport import RawKVTransportStats, build_rosetta_transport
from transformers.utils import ModelOutput
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None


def _safe_is_gradient_checkpointing_enabled(model: nn.Module) -> bool:
        """
        某些 transformers 模型在访问 is_gradient_checkpointing 时会进一步遍历
        named_modules()；在我们当前的训练/评测路径里，这一步偶发抛异常。
        这里做保护性探测，失败时退回 False，避免评测阶段被非核心状态查询打断。
        """
        try:
            return bool(getattr(model, "is_gradient_checkpointing", False))
        except Exception:
            return False


def _safe_toggle_gradient_checkpointing(model: nn.Module, enabled: bool) -> bool:
        """
        尝试安全地开关 gradient checkpointing。
        返回值表示这次切换是否真的成功，便于 finally 中只恢复成功关闭过的状态。
        """
        method_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        toggle_fn = getattr(model, method_name, None)
        if not callable(toggle_fn):
            return False

        try:
            toggle_fn()
            return True
        except Exception:
            return False

def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
        new_cache = DynamicCache()
        for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
            new_cache.key_cache.append(k.clone().detach())
            new_cache.value_cache.append(v.clone().detach())
        return new_cache

def hybrid_to_dynamic(hybrid_cache):
    if hybrid_cache is None:
        return None
    if isinstance(hybrid_cache, DynamicCache):
        return hybrid_cache

    # 手动从 HybridCache 提取
    if hasattr(hybrid_cache, "key_cache") and hasattr(hybrid_cache, "value_cache"):
        keys = hybrid_cache.key_cache
        values = hybrid_cache.value_cache
        assert len(keys) == len(values), "key/value 层数不一致"

        legacy_cache = [(k, v) for k, v in zip(keys, values)]
        return DynamicCache.from_legacy_cache(legacy_cache)

    raise TypeError(f"Unsupported cache type: {type(hybrid_cache)}")

class RosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM
    """
    def __init__(self, model_list: List[PreTrainedModel], base_model_idx = 0, projector_list: List[Projector] = [], include_response: bool = False, multi_source_fusion_mode: str = "parallel", adaptive_quant_table: Optional[AdaptiveCoefficientQuantizer] = None, cache_alignment: str = "fuser", transport_config: Optional[dict] = None):
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)
        self.adaptive_quant_table = adaptive_quant_table
        if self.adaptive_quant_table is not None:
            # Keep allocator/entropy parameters in FP32 while allowing its output
            # cache to return to the receiver cache dtype.
            self.adaptive_quant_table.to(device=device)
        self.adaptive_quant_table_step = 0
        self.last_adaptive_quant_table_metrics = None
        self.transport = build_rosetta_transport(transport_config)
        self.last_transport_stats: Optional[RawKVTransportStats] = None

        self.projector_dict = {}
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []
        self.last_fusion_stats = None
        if cache_alignment not in {"fuser", "concat"}:
            raise ValueError("cache_alignment must be 'fuser' or 'concat'.")
        self.cache_alignment = cache_alignment

        # Multi-source fusion mode: 
        # "sequential" (default): each source updates base cache iteratively
        # "parallel": all sources project from clean base cache, then sum projections
        self.include_response = include_response
        if multi_source_fusion_mode not in ["sequential", "parallel"]:
            raise ValueError(f"multi_source_fusion_mode must be 'sequential' or 'parallel', got '{multi_source_fusion_mode}'")
        self.multi_source_fusion_mode = multi_source_fusion_mode

    @staticmethod
    def _raw_cache_to_wire(cache: DynamicCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Move the uncompressed sharer cache to a serializable CPU wire payload."""
        return tuple(
            (
                key.detach().to(device="cpu").contiguous(),
                value.detach().to(device="cpu").contiguous(),
            )
            for key, value in zip(cache.key_cache, cache.value_cache)
        )

    @staticmethod
    def _raw_cache_from_wire(
        wire_payload: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        device: torch.device,
    ) -> DynamicCache:
        restored = DynamicCache()
        for key, value in wire_payload:
            restored.key_cache.append(key.to(device=device, non_blocking=False))
            restored.value_cache.append(value.to(device=device, non_blocking=False))
        return restored

    def _accumulate_transport_stats(self, stats: RawKVTransportStats) -> None:
        previous = self.last_transport_stats
        if previous is None:
            self.last_transport_stats = stats
            return
        self.last_transport_stats = RawKVTransportStats(
            payload_bytes=previous.payload_bytes + stats.payload_bytes,
            serialize_seconds=previous.serialize_seconds + stats.serialize_seconds,
            transmit_seconds=previous.transmit_seconds + stats.transmit_seconds,
            deserialize_seconds=previous.deserialize_seconds + stats.deserialize_seconds,
        )

    def _transport_raw_cache(self, cache: DynamicCache, device: torch.device) -> DynamicCache:
        if self.transport is None:
            return cache
        restored_payload = self.transport.roundtrip(self._raw_cache_to_wire(cache))
        if self.transport.last_stats is None:
            raise RuntimeError("Rosetta raw-KV transport completed without transport statistics.")
        self._accumulate_transport_stats(self.transport.last_stats)
        return self._raw_cache_from_wire(restored_payload, device)

    def update_adaptive_quant_table_schedule(self, step: int) -> None:
        self.adaptive_quant_table_step = max(0, int(step))
        if self.adaptive_quant_table is not None:
            self.adaptive_quant_table.update_temperature(self.adaptive_quant_table_step)

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device
    
    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        """
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        return self
        
    # set projector 
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        Set the projector configuration
        Args:
            source_model_idx: int, the index of the source model
            source_model_layer_idx: int, the index of the source model layer
            target_model_idx: int, the index of the target model
            target_model_layer_idx: int, the index of the target model layer
            projector_idx: int, the index of the projector

        The projector dict structure supports multiple projectors per target layer.
        Structure:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))


    def load_projector(self, projector_list):
        self.projector_list: List[Projector] = projector_list

    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx):
        pair_list = self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        return self.projector_list[pair_list[0][1]]

    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj
    
    
    def save_projector_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    
    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

    @staticmethod
    def _monkeypatch_qwen3_attention_forward(attn_module, new_k_cache, new_v_cache):
        """
        Monkeypatch Qwen3Attention.forward so that *current step* attention uses the
        provided key/value (in cache space) before computing attention.

        This avoids editing transformers' Qwen3 code while ensuring the modified KV
        is used in the same forward pass (not just for the next token).

        new_k_cache/new_v_cache: (B, kv_heads, q_len, head_dim) in the SAME space as
        Qwen3Attention's key_states/value_states AFTER k_norm + RoPE (k) and reshape (v).
        """
        import types

        # Lazy imports to avoid hard dependency at module import time
        from transformers.models.qwen3.modeling_qwen3 import (  # type: ignore
            apply_rotary_pos_emb,
            eager_attention_forward,
            ALL_ATTENTION_FUNCTIONS,
        )

        orig_forward = attn_module.forward

        def patched_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings,
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            # This is essentially Qwen3Attention.forward with one injection point.
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # === Injection point (before cache update & attention) ===
            # Replace current-token key/value with provided cache-space tensors.
            # Expect same shape as key_states/value_states at this moment:
            # (B, kv_heads, q_len, head_dim)
            if new_k_cache is not None and new_v_cache is not None:
                # Only replace if compatible
                if key_states.shape == new_k_cache.shape:
                    key_states = new_k_cache
                if value_states.shape == new_v_cache.shape:
                    value_states = new_v_cache

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            attention_interface = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    # fall back to eager, same as upstream behavior (warning omitted here)
                    attention_interface = eager_attention_forward
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        attn_module.forward = types.MethodType(patched_forward, attn_module)
        return orig_forward

    def register_hooks(self, input_ids, attention_mask, position_ids, base_kv_cache, source_model_idx, source_kv_cache):

        base_kv_copy = clone_kv_cache(base_kv_cache)
        source_kv_copy = clone_kv_cache(source_kv_cache)

        new_length = input_ids.shape[1]

        base_output_kv_cache = self.model_list[self.base_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=base_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values
        source_output_kv_cache = self.model_list[source_model_idx].forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask, 
                    position_ids=position_ids,
                    past_key_values=source_kv_copy,
                    labels=None,
                    use_cache=True, 
                ).past_key_values        
        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            pair_list = entry

            projected_kv_list = []
            source_kv_list = []
            for source_model_layer_idx, projector_idx in pair_list:
                source_key_cache, source_value_cache = source_output_kv_cache[source_model_layer_idx]
                new_source_key_cache = source_key_cache[:, :, -new_length:, :]
                new_source_value_cache = source_value_cache[:, :, -new_length:, :]
                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                projected_key, projected_value = self.projector_list[projector_idx].forward(
                    new_source_kv_cache, # tuple of (key, value), each of shape (B, N, H, D)
                    new_base_kv_cache
                )
                projected_kv_list.append((projected_key, projected_value))
                source_kv_list.append(new_source_kv_cache)

            # Use first projector result
            agg_key, agg_value = projected_kv_list[0]

            # Update cache
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][:, :, -new_length:, :] = agg_value

        # Monkeypatch attention forward so the modified KV is used in *this* forward pass.
        hook_handlers = []  # list of (attn_module, orig_forward)
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache
    
    def remove_hooks(self, hook_handlers):
        # Restore monkeypatched forwards
        for attn, orig_forward in hook_handlers:
            attn.forward = orig_forward

    def _forward_concat_lcf_first(
        self,
        input_ids,
        attention_mask,
        labels,
        sharer_prompt_only: bool = False,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Train LCF-first by prepending a decoded Sharer cache to Receiver.

        The Sharer only sees the prompt.  Its keys are captured before RoPE,
        each routed layer is compressed and expanded by an LCFFirstProjector,
        and Receiver RoPE is then applied at compact prefix positions.  The
        frozen Receiver supplies the language-model loss and its gradient flows
        back through the prefix cache into the LCF modules.
        """
        from rosetta.cachejpeg_rosetta.pre_rope import (
            apply_receiver_compact_rope,
            capture_pre_rope_keys,
            replace_cache_keys_with_pre_rope,
        )
        from rosetta.model.latent_kv import LCFFirstProjector
        from rosetta.model.lcf_projected_kv import LCFProjectedKVProjector
        from rosetta.model.direct_pre_rope_mlp import DirectPreRopeMLPProjector

        if not isinstance(input_ids, list) or not isinstance(attention_mask, list):
            raise ValueError(
                "concat LCF-first training requires per-model input_ids and "
                "attention_mask lists. Use independent dual-tokenizer inputs or "
                "set model.is_do_alignment=true."
            )
        if len(self.model_list) != 2:
            raise ValueError("concat LCF-first training currently requires one Receiver and one Sharer.")
        base_ids = input_ids[self.base_model_idx]
        base_mask = attention_mask[self.base_model_idx]
        sharer_idx = 1 if self.base_model_idx == 0 else 0
        sharer_ids = input_ids[sharer_idx]
        sharer_mask = attention_mask[sharer_idx]
        if labels is None:
            raise ValueError("concat LCF-first training requires labels to locate the prompt boundary.")
        if base_ids.shape[0] != 1:
            raise ValueError(
                "concat LCF-first currently requires per_device_train_batch_size=1 "
                "because prompt boundaries may differ within a batch."
            )
        if torch.is_tensor(sharer_prompt_only):
            sharer_prompt_only = bool(sharer_prompt_only.flatten()[0].item())
        supervised = (labels[0] != -100).nonzero(as_tuple=False)
        if supervised.numel() == 0:
            raise ValueError("Training sample has no supervised response tokens.")
        receiver_prompt_length = int(supervised[0].item())
        if receiver_prompt_length <= 0:
            raise ValueError("Training sample has an empty Receiver prompt prefix.")

        if sharer_prompt_only:
            sharer_prompt_ids = sharer_ids
            sharer_prompt_mask = sharer_mask
        else:
            # Backward-compatible aligned input path. Both token streams share
            # a padded positional grid, so the Receiver label boundary can be
            # used to slice the Sharer prompt.
            sharer_prompt_ids = sharer_ids[:, :receiver_prompt_length]
            sharer_prompt_mask = sharer_mask[:, :receiver_prompt_length]
        if not bool(sharer_prompt_mask.bool().any()):
            raise ValueError("Training sample has an empty Sharer prompt prefix.")

        sharer_model = self.model_list[sharer_idx]
        with torch.no_grad(), capture_pre_rope_keys(sharer_model) as captured:
            sharer_output = sharer_model(
                input_ids=sharer_prompt_ids,
                attention_mask=sharer_prompt_mask,
                use_cache=True,
                return_dict=True,
            )
        sharer_cache = replace_cache_keys_with_pre_rope(
            sharer_output.past_key_values, captured
        )

        routing = self.projector_dict.get(self.base_model_idx, {}).get(sharer_idx, {})
        receiver_layers = self.model_list[self.base_model_idx].config.num_hidden_layers
        if set(routing) != set(range(receiver_layers)):
            raise ValueError("concat LCF-first requires exactly one route for every Receiver layer.")
        encoded_layers = []
        for receiver_layer in range(receiver_layers):
            pairs = routing[receiver_layer]
            if len(pairs) != 1:
                raise ValueError("concat LCF-first requires one Sharer layer per Receiver layer.")
            source_layer, projector_idx = pairs[0]
            projector = self.projector_list[projector_idx]
            if not isinstance(
                projector,
                (
                    LCFFirstProjector,
                    LCFProjectedKVProjector,
                    DirectPreRopeMLPProjector,
                ),
            ):
                raise TypeError(
                    "concat cache_alignment requires LCFFirstProjector or "
                    "LCFProjectedKVProjector or DirectPreRopeMLPProjector modules."
                )
            # Projectors and Sharer cache are placed on the Rosetta model device
            # during setup. Avoid walking the projector parameter tree on every
            # sample merely to rediscover that device; repeated module introspection
            # is unnecessary here and can be fragile after hooks.
            projector_device = sharer_cache.key_cache[source_layer].device
            source_kv = (
                sharer_cache.key_cache[source_layer].to(projector_device),
                sharer_cache.value_cache[source_layer].to(projector_device),
            )
            if isinstance(projector, DirectPreRopeMLPProjector):
                receiver_key, receiver_value = projector.project(source_kv)
                encoded_layers.append(
                    (projector, receiver_key, receiver_value)
                )
                continue
            if isinstance(projector, LCFProjectedKVProjector):
                key_latent, value_latent = projector.encode(source_kv)
            else:
                latent = projector.encode(source_kv)  # Existing direct split path.
                key_latent, value_latent = latent.chunk(2, dim=-1)
            encoded_layers.append((projector, key_latent, value_latent))

        quant_result = None
        if self.adaptive_quant_table is not None:
            if any(
                isinstance(projector, DirectPreRopeMLPProjector)
                for projector, _key, _value in encoded_layers
            ):
                raise ValueError(
                    "Direct pre-RoPE MLP concat does not support adaptive quantization."
                )
            # The wire representation is the LCF transport K/V, not the
            # original Sharer KV. These are either the legacy direct halves or
            # the learned projected views, both represented with one KV head.
            latent_wire_cache = tuple(
                (
                    key_latent.unsqueeze(1),
                    value_latent.unsqueeze(1),
                )
                for _projector, key_latent, value_latent in encoded_layers
            )
            quant_result = self.adaptive_quant_table(latent_wire_cache)

        prefix = DynamicCache()
        for receiver_layer, (projector, key_latent, value_latent) in enumerate(encoded_layers):
            if quant_result is not None:
                quantized_key, quantized_value = quant_result.past_key_values[receiver_layer]
                key_latent = quantized_key.squeeze(1)
                value_latent = quantized_value.squeeze(1)
            if isinstance(projector, DirectPreRopeMLPProjector):
                receiver_key, receiver_value = key_latent, value_latent
            elif isinstance(projector, LCFProjectedKVProjector):
                receiver_key, receiver_value = projector.decode_transport(
                    key_latent, value_latent
                )
            else:
                receiver_key, receiver_value = projector.decode(
                    torch.cat((key_latent, value_latent), dim=-1)
                )
            prefix.key_cache.append(receiver_key)
            prefix.value_cache.append(receiver_value)

        receiver_model = self.model_list[self.base_model_idx]
        prefix = apply_receiver_compact_rope(receiver_model, prefix)
        prefix_length = prefix.key_cache[0].shape[-2]
        prefix_mask = sharer_prompt_mask.to(base_mask.device)
        combined_mask = torch.cat((prefix_mask, base_mask), dim=-1)
        position_ids = (
            base_mask.long().cumsum(-1) - 1 + prefix_length
        ).masked_fill(base_mask == 0, 0)
        cache_position = torch.arange(
            prefix_length,
            prefix_length + base_ids.shape[1],
            device=base_ids.device,
        )
        output = receiver_model(
            input_ids=base_ids,
            attention_mask=combined_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=prefix,
            labels=labels,
            use_cache=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        if quant_result is not None:
            rate_weight = self.adaptive_quant_table.rate_weight(
                self.adaptive_quant_table_step
            )
            rate_loss = quant_result.estimated_payload_bits * rate_weight
            if output.loss is not None:
                output.loss = output.loss + rate_loss
            self.last_adaptive_quant_table_metrics = {
                "estimated_payload_bits": float(
                    quant_result.estimated_payload_bits.detach().item()
                ),
                "rate_source": "lcf_latent_pre_decode",
                "rate_weight": float(rate_weight),
                "rate_loss": float(rate_loss.detach().item()),
                "temperature": float(
                    self.adaptive_quant_table.gumbel_temperature.item()
                ),
                "mean_alpha": float(
                    quant_result.alpha.detach().float().mean().item()
                ),
            }
        else:
            self.last_adaptive_quant_table_metrics = None
        return output

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        sharer_prompt_only: bool = False,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # **kwargs: Unpack[KwargsForCausalLM],
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass
        
        kv_cache_index: List of tensors with shape (B, sec_seq_len, 2).
            The first element [i][0][0][0] controls sharer selection:
            - -1: No projection (receiver only, skip all sharers)
            - 0: Self projection (receiver projects from itself) - not currently used
            - >0: Bitmask selecting sharers (1 (001)=sharer1, 2 (010)=sharer2, 3 (011)=both, 7 (111)=all three)
            Each bit corresponds to a sharer: bit i selects sharer at model_list[i+1].
        
        input_ids: If LongTensor, same input for all models. If List, per-model inputs.
        """

        if self.cache_alignment == "concat":
            return self._forward_concat_lcf_first(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                sharer_prompt_only=sharer_prompt_only,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )

        # Handle different input formats: if input_ids is a list, use per-model inputs
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()
            self.last_transport_stats = None
            
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        curr_base_kv_cache = past_key_values
        total_adaptive_payload_bits = None

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:

                if self.include_response:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(input_ids=prefill_input_ids, attention_mask=prefill_attention_mask, position_ids=prefill_position_ids,
                                                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                                                        source_model_idx=1, 
                                                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][1])

                # calculate target model kvcache
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=True, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                if self.include_response:
                    self.remove_hooks(hook_handlers)

                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(base_output_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(source_output_kv_cache)

            else:

                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask, 
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=use_cache, 
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

                curr_base_kv_cache: DynamicCache = output.past_key_values
            
                for source_model_idx in range(1, len(self.model_list)):
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    if isinstance(input_ids, list):
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    model = self.model_list[source_model_idx]
                    was_training = model.training
                    had_gc = _safe_is_gradient_checkpointing_enabled(model)
                    disabled_gc = False

                    try:
                        if was_training:
                            model.eval()
                        if had_gc:
                            disabled_gc = _safe_toggle_gradient_checkpointing(model, enabled=False)

                        with torch.no_grad():
                            out = model(
                                input_ids=source_prefill_input_ids,
                                attention_mask=source_prefill_attention_mask,
                                position_ids=prefill_position_ids,
                                past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                                use_cache=True,
                                return_dict=True,
                            )
                            curr_source_kv_cache = out.past_key_values
                    finally:
                        if disabled_gc:
                            _safe_toggle_gradient_checkpointing(model, enabled=True)
                        if was_training:
                            model.train()
                    
                    curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(curr_source_kv_cache)

                # Send each selected sharer's complete, uncompressed KV cache
                # through one serial transport boundary before fuser projection.
                # This intentionally performs no layer-level parallelism.
                if self.base_model_idx in self.projector_dict:
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0 and self.transport is not None:
                        for source_model_idx in self.projector_dict[self.base_model_idx]:
                            if sharer_mask & (1 << (source_model_idx - 1)):
                                source_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx]
                                self.kv_cache_dict[self.base_model_idx][source_model_idx] = (
                                    self._transport_raw_cache(
                                        source_cache,
                                        device=source_prefill_input_ids.device,
                                    )
                                )

                # calculate source model kvcache and apply projections
                if self.base_model_idx in self.projector_dict:
                    # Iterate over all source models in projector_dict
                    sharer_mask = kv_cache_index[i][0][0][0].item()
                    if sharer_mask > 0:
                        base_cache = clone_kv_cache(curr_base_kv_cache)

                        # For parallel mode, accumulate residuals for each target layer
                        parallel_delta_cache = {} if self.multi_source_fusion_mode == "parallel" else None
                        
                        # Compute and apply projections (shared logic for both modes)
                        for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                            # Check if this sharer is selected: bit (source_model_idx - 1)
                            if not (sharer_mask & (1 << (source_model_idx - 1))):
                                continue
                            if self.multi_source_fusion_mode == "sequential":
                                base_cache_ref = curr_base_kv_cache
                            else:
                                # Parallel: always project from the clean cloned base cache
                                base_cache_ref = base_cache

                            quantized_source_by_target = None
                            if self.adaptive_quant_table is not None:
                                # R_hat is estimated from routed sharer KV before the
                                # split projector sees receiver KV or performs fusion.
                                expected_layers = list(range(self.adaptive_quant_table.num_layers))
                                layer_map = self.projector_dict[self.base_model_idx][source_model_idx]
                                actual_layers = sorted(int(layer_idx) for layer_idx in layer_map)
                                if actual_layers != expected_layers:
                                    raise ValueError(
                                        "adaptive_quant_table requires one routed source cache for "
                                        f"every receiver layer; expected {expected_layers}, got {actual_layers}."
                                    )
                                routed_source = []
                                for target_layer_idx in expected_layers:
                                    pair_list = layer_map[target_layer_idx]
                                    if len(pair_list) != 1:
                                        raise ValueError(
                                            "Pre-projector adaptive quantization requires exactly one "
                                            "source layer per receiver layer."
                                        )
                                    source_layer_idx, _ = pair_list[0]
                                    source_key, source_value = self.kv_cache_dict[
                                        self.base_model_idx
                                    ][source_model_idx][source_layer_idx]
                                    routed_source.append(
                                        (
                                            source_key[:, :, start:end, :],
                                            source_value[:, :, start:end, :],
                                        )
                                    )
                                quantized_source = self.adaptive_quant_table(tuple(routed_source))
                                quantized_source_by_target = {
                                    layer_idx: reconstructed
                                    for layer_idx, reconstructed in enumerate(
                                        quantized_source.past_key_values
                                    )
                                }
                                total_adaptive_payload_bits = (
                                    quantized_source.estimated_payload_bits
                                    if total_adaptive_payload_bits is None
                                    else total_adaptive_payload_bits
                                    + quantized_source.estimated_payload_bits
                                )

                            projected_by_layer = {}
                            for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                                # Get base KV cache slice for projection
                                base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                                new_base_key_cache = base_key_cache[:, :, start:end, :]
                                new_base_value_cache = base_value_cache[:, :, start:end, :]
                                new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                                pair_list = entry

                                projected_kv_list = []
                                source_kv_list = []
                                for source_model_layer_idx, projector_idx in pair_list:
                                    if quantized_source_by_target is not None:
                                        new_source_kv_cache = quantized_source_by_target[
                                            int(target_layer_idx)
                                        ]
                                    else:
                                        source_key_cache, source_value_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx][source_model_layer_idx]
                                        new_source_kv_cache = (
                                            source_key_cache[:, :, start:end, :],
                                            source_value_cache[:, :, start:end, :],
                                        )
                                    projected_key, projected_value = self.projector_list[projector_idx].forward(
                                        new_source_kv_cache,
                                        new_base_kv_cache
                                    )
                                    projected_kv_list.append((projected_key, projected_value))
                                    source_kv_list.append(new_source_kv_cache)

                                # Use first projector result
                                agg_key, agg_value = projected_kv_list[0]

                                projected_by_layer[int(target_layer_idx)] = (
                                    (agg_key, agg_value),
                                    new_base_kv_cache,
                                )

                            for target_layer_idx, (
                                (agg_key, agg_value),
                                new_base_kv_cache,
                            ) in projected_by_layer.items():
                                # Collect or apply projection based on mode
                                if self.multi_source_fusion_mode == "sequential":
                                    # Sequential: apply immediately so next source sees updated cache
                                    curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                    curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                                else:
                                    # Parallel: accumulate residuals (agg - base) for this target layer
                                    if target_layer_idx not in parallel_delta_cache:
                                        parallel_delta_cache[target_layer_idx] = (
                                            torch.zeros_like(new_base_key_cache),
                                            torch.zeros_like(new_base_value_cache),
                                        )
                                    delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                    delta_key = delta_key + (agg_key - new_base_key_cache)
                                    delta_value = delta_value + (agg_value - new_base_value_cache)
                                    parallel_delta_cache[target_layer_idx] = (delta_key, delta_value)

                        # For parallel mode, apply all accumulated residuals in one shot
                        if self.multi_source_fusion_mode == "parallel":
                            for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                                base_key_cache, base_value_cache = base_cache[target_layer_idx]
                                base_key_slice = base_key_cache[:, :, start:end, :]
                                base_value_slice = base_value_cache[:, :, start:end, :]
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = base_key_slice + delta_key
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = base_value_slice + delta_value

                output.past_key_values = curr_base_kv_cache

        if self.adaptive_quant_table is not None and total_adaptive_payload_bits is not None:
            rate_weight = self.adaptive_quant_table.rate_weight(
                self.adaptive_quant_table_step
            )
            rate_loss = total_adaptive_payload_bits * rate_weight
            if output.loss is not None:
                output.loss = output.loss + rate_loss
            result = self.adaptive_quant_table.last_result
            self.last_adaptive_quant_table_metrics = {
                "estimated_payload_bits": float(
                    total_adaptive_payload_bits.detach().item()
                ),
                "rate_source": "sharer_raw_kv_pre_fusion",
                "rate_weight": float(rate_weight),
                "rate_loss": float(rate_loss.detach().item()),
                "temperature": float(
                    self.adaptive_quant_table.gumbel_temperature.item()
                ),
                "mean_alpha": (
                    float(result.alpha.detach().float().mean().item())
                    if result is not None
                    else None
                ),
            }
        else:
            self.last_adaptive_quant_table_metrics = None
                                                                             
        latent_layer_stats = []
        for projector_idx, projector in enumerate(self.projector_list):
            projector_stats = getattr(projector, "last_stats", None)
            if projector_stats is not None:
                latent_layer_stats.append(
                    {"projector_idx": projector_idx, **dict(projector_stats)}
                )
        if latent_layer_stats:
            first_projector = self.projector_list[0]
            fusion_type = (
                "latent_kv_split"
                if first_projector.__class__.__name__ == "SplitLatentKVProjector"
                else "latent_kv_joint"
            )
            self.last_fusion_stats = {
                "fusion_type": fusion_type,
                "latent_dim": getattr(first_projector, "latent_dim", None),
                "joint_input_dim": getattr(first_projector, "joint_input_dim", None),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for projector in self.projector_list
                    for parameter in projector.parameters()
                    if parameter.requires_grad
                ),
                "layers": latent_layer_stats,
            }
        else:
            self.last_fusion_stats = None

        return output
    
    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        streamer = None,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """

        self.kv_cache_dict = dict()

        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        # Resolve base inputs
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)

        # Prefill to build caches and obtain initial logits
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        # Initialize streamer with prompt if provided
        if streamer is not None:
            streamer.put(base_input_ids)

        # EOS handling setup
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # Apply repetition/presence/frequency penalties to logits before sampling
            adjusted_logits = last_logits
            if (
                (repetition_penalty is not None and repetition_penalty != 1.0) or
                (presence_penalty is not None and presence_penalty != 0.0) or
                (frequency_penalty is not None and frequency_penalty != 0.0)
            ):
                adjusted_logits = last_logits.clone()
                vocab_size = adjusted_logits.size(-1)
                # Per-batch penalty application for clarity and correctness
                for b in range(batch_size):
                    seq_tokens = all_input_ids[b]
                    if seq_tokens.numel() == 0:
                        continue
                    counts = torch.bincount(seq_tokens, minlength=vocab_size)
                    if counts.dtype != torch.float32 and counts.dtype != torch.float64:
                        counts = counts.to(adjusted_logits.dtype)
                    # Presence penalty: penalize any token that has appeared
                    if presence_penalty and presence_penalty != 0.0:
                        presence_mask = counts > 0
                        if presence_mask.any():
                            adjusted_logits[b, presence_mask] = adjusted_logits[b, presence_mask] - presence_penalty
                    # Frequency penalty: penalize proportionally to frequency
                    if frequency_penalty and frequency_penalty != 0.0:
                        adjusted_logits[b] = adjusted_logits[b] - frequency_penalty * counts
                    # Repetition penalty (HF-style): divide positive logits, multiply negative logits
                    if repetition_penalty and repetition_penalty != 1.0:
                        rep_mask = counts > 0
                        if rep_mask.any():
                            pos_mask = rep_mask & (adjusted_logits[b] > 0)
                            neg_mask = rep_mask & ~pos_mask
                            if pos_mask.any():
                                adjusted_logits[b, pos_mask] = adjusted_logits[b, pos_mask] / repetition_penalty
                            if neg_mask.any():
                                adjusted_logits[b, neg_mask] = adjusted_logits[b, neg_mask] * repetition_penalty

            # Sample next token
            next_token = sample_token(adjusted_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # Apply EOS logic
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Stream the new token if streamer provided
            if streamer is not None:
                streamer.put(next_token_unsqueezed)

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            last_logits = decode_output.logits[:, -1, :]

        # End streaming if streamer provided
        if streamer is not None:
            streamer.end()

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids
