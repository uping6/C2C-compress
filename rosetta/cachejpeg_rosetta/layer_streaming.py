from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers.cache_utils import DynamicCache

from rosetta.cachejpeg.transport import TransportStats, serialize_payload


@dataclass(frozen=True)
class LayerTask:
    layer_idx: int
    key: torch.Tensor
    value: torch.Tensor
    ready_event: torch.cuda.Event | None = None


class StreamingDynamicCache(DynamicCache):
    """DynamicCache that publishes each layer after its first prefill update."""

    def __init__(self, on_layer_ready: Callable[[LayerTask], None]):
        super().__init__()
        self._on_layer_ready = on_layer_ready
        self._published_layers: set[int] = set()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key, value = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx not in self._published_layers:
            self._published_layers.add(int(layer_idx))
            ready_event = None
            if key.is_cuda:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(key.device))
            self._on_layer_ready(
                LayerTask(int(layer_idx), key.detach(), value.detach(), ready_event)
            )
        return key, value


class LayerCompressionPipeline:
    """Bounded producer/consumer pipeline for encode, transfer, and decode."""

    _STOP = object()

    def __init__(
        self,
        *,
        codec: Any,
        codec_config: dict[str, Any],
        transport: Any | None,
        num_layers: int,
        queue_size: int = 2,
    ):
        if not hasattr(codec, "encode_layer"):
            raise TypeError("Layer streaming requires a codec with encode_layer().")
        self.codec = codec
        self.codec_config = codec_config
        self.transport = transport
        self.num_layers = int(num_layers)
        self.tasks: queue.Queue[LayerTask | object] = queue.Queue(maxsize=max(1, int(queue_size)))
        self.layers: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        self.payload_bytes = 0
        self.original_kv_bytes = 0
        self.encode_seconds = 0.0
        self.layer_encode_seconds: list[float | None] = [None] * self.num_layers
        self.decode_seconds = 0.0
        self.transport_stats: list[TransportStats] = []
        self.error: BaseException | None = None
        self._cuda_stream = (
            torch.cuda.Stream(device=codec.device)
            if getattr(codec, "device", torch.device("cpu")).type == "cuda"
            else None
        )
        self._worker = threading.Thread(
            target=self._run, name="cachejpeg-layer-worker", daemon=True
        )
        self._worker.start()

    def submit(self, task: LayerTask) -> None:
        while True:
            if self.error is not None:
                raise RuntimeError("CacheJPEG layer worker failed.") from self.error
            try:
                self.tasks.put(task, timeout=0.1)
                return
            except queue.Full:
                continue

    def _process(self, task: LayerTask) -> None:
        self.original_kv_bytes += int(
            task.key.numel() * task.key.element_size()
            + task.value.numel() * task.value.element_size()
        )
        stream_context = (
            torch.cuda.stream(self._cuda_stream)
            if self._cuda_stream is not None
            else nullcontext()
        )
        with stream_context:
            if self._cuda_stream is not None and task.ready_event is not None:
                self._cuda_stream.wait_event(task.ready_event)
            started = time.perf_counter()
            payload = self.codec.encode_layer(
                task.layer_idx,
                self.num_layers,
                task.key,
                task.value,
                self.codec_config,
            )
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()
            layer_encode_seconds = time.perf_counter() - started
            self.encode_seconds += layer_encode_seconds
            self.layer_encode_seconds[task.layer_idx] = layer_encode_seconds

            self.payload_bytes += len(serialize_payload(payload))
            received = self.transport.roundtrip(payload) if self.transport is not None else payload
            if self.transport is not None and self.transport.last_stats is not None:
                self.transport_stats.append(self.transport.last_stats)

            started = time.perf_counter()
            decoded = self.codec.decode(received, self.codec_config)
            if self._cuda_stream is not None:
                self._cuda_stream.synchronize()
            self.decode_seconds += time.perf_counter() - started
        if len(decoded) != 1:
            raise ValueError("A streamed CacheJPEG layer payload must decode to exactly one layer.")
        if self.layers[task.layer_idx] is not None:
            raise ValueError(f"Duplicate streamed CacheJPEG layer {task.layer_idx}.")
        self.layers[task.layer_idx] = decoded[0]

    def _run(self) -> None:
        try:
            while True:
                item = self.tasks.get()
                try:
                    if item is self._STOP:
                        return
                    self._process(item)
                finally:
                    self.tasks.task_done()
        except BaseException as exc:
            self.error = exc

    def finish(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        self.submit_stop()
        self._worker.join()
        if self.error is not None:
            raise RuntimeError("CacheJPEG layer worker failed.") from self.error
        missing = [idx for idx, layer in enumerate(self.layers) if layer is None]
        if missing:
            raise ValueError(f"Sharer did not publish CacheJPEG layers: {missing}")
        return tuple(layer for layer in self.layers if layer is not None)

    def submit_stop(self) -> None:
        while self._worker.is_alive():
            try:
                self.tasks.put(self._STOP, timeout=0.1)
                return
            except queue.Full:
                if self.error is not None:
                    return

    def abort(self) -> None:
        self.submit_stop()
        self._worker.join()

    def aggregate_transport_stats(self) -> TransportStats | None:
        if not self.transport_stats:
            return None
        return TransportStats(
            payload_bytes=sum(item.payload_bytes for item in self.transport_stats),
            serialize_seconds=sum(item.serialize_seconds for item in self.transport_stats),
            transmit_seconds=sum(item.transmit_seconds for item in self.transport_stats),
            deserialize_seconds=sum(item.deserialize_seconds for item in self.transport_stats),
        )


class LayerPrefillTimer:
    """Measure each decoder layer's forward time with CUDA events or wall time."""

    def __init__(self, model: torch.nn.Module, num_layers: int):
        decoder = getattr(model, "model", model)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            transformer = getattr(model, "transformer", None)
            layers = getattr(transformer, "h", None) if transformer is not None else None
        if layers is None or len(layers) != int(num_layers):
            raise ValueError("Unable to locate sharer decoder layers for per-layer prefill timing.")
        self.layers = layers
        self.num_layers = int(num_layers)
        parameter = next(model.parameters())
        self.use_cuda = parameter.device.type == "cuda"
        self.starts: list[Any | None] = [None] * self.num_layers
        self.ends: list[Any | None] = [None] * self.num_layers
        self.handles = []

    def _pre_hook(self, layer_idx: int):
        def hook(_module, _inputs):
            if self.use_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.starts[layer_idx] = event
            else:
                self.starts[layer_idx] = time.perf_counter()
        return hook

    def _post_hook(self, layer_idx: int):
        def hook(_module, _inputs, _output):
            if self.use_cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.ends[layer_idx] = event
            else:
                self.ends[layer_idx] = time.perf_counter()
        return hook

    def start(self) -> None:
        for layer_idx, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(self._pre_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(self._post_hook(layer_idx)))

    def finish(self) -> list[float]:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        durations = []
        for layer_idx, (started, ended) in enumerate(zip(self.starts, self.ends)):
            if started is None or ended is None:
                raise ValueError(f"Missing sharer prefill timing for layer {layer_idx}.")
            if self.use_cuda:
                durations.append(float(started.elapsed_time(ended)) / 1000.0)
            else:
                durations.append(float(ended - started))
        return durations
