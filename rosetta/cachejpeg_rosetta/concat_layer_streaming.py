from __future__ import annotations

import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from rosetta.cachejpeg.transport import TransportStats, serialize_payload

from .cache_aligner import ConcatCacheAligner, LCFLatentRouting
from .pre_rope import PreRopeLayerTask


@dataclass(frozen=True)
class _EncodedLayer:
    route: tuple[int, int, int]
    payload: Any


@dataclass(frozen=True)
class _ReceivedLayer:
    route: tuple[int, int, int]
    payload: Any


class ConcatLayerPipeline:
    """Pipeline pre-RoPE layers through LCF, CacheJPEG, transport and LCF-up."""

    _STOP = object()

    def __init__(
        self,
        *,
        aligner: ConcatCacheAligner,
        codec: Any,
        codec_config: dict[str, Any],
        transport: Any | None,
        routing: LCFLatentRouting,
        gpu_streams: int = 2,
        max_inflight_layers: int = 4,
        zero_sharer_cache_at_receiver: bool = False,
    ) -> None:
        if not hasattr(codec, "encode_layer"):
            raise TypeError("Concat layer streaming requires a GPU codec with encode_layer().")
        self.aligner = aligner
        self.codec = codec
        self.codec_config = codec_config
        self.transport = transport
        self.routing = routing
        self.num_layers = len(routing.routes)
        self.route_by_source = {route[1]: route for route in routing.routes}
        self.worker_count = max(1, min(int(gpu_streams), self.num_layers))
        self.max_inflight_layers = max(1, int(max_inflight_layers))
        self.zero_sharer_cache_at_receiver = bool(zero_sharer_cache_at_receiver)

        self.encode_queue: queue.Queue[Any] = queue.Queue()
        self.transport_queue: queue.Queue[Any] = queue.Queue()
        self.decode_queue: queue.Queue[Any] = queue.Queue()
        self.inflight = threading.Semaphore(self.max_inflight_layers)
        self.lock = threading.Lock()
        self.error: BaseException | None = None
        self.decoded_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.transport_stats: list[TransportStats] = []

        self.original_kv_bytes = 0
        self.latent_kv_bytes = 0
        self.payload_bytes = 0
        self.lcf_encode_seconds = 0.0
        self.codec_encode_seconds = 0.0
        self.codec_decode_seconds = 0.0
        self.lcf_decode_seconds = 0.0
        self.layer_timings: dict[int, dict[str, float]] = {}
        self.stage_intervals: dict[str, list[tuple[float, float]]] = {
            "lcf_encode": [],
            "encode": [],
            "decode": [],
            "lcf_decode": [],
        }
        self.started_at = time.perf_counter()

        device = getattr(codec, "device", torch.device("cpu"))
        self.encode_streams = [
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
            for _ in range(self.worker_count)
        ]
        self.decode_streams = [
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
            for _ in range(self.worker_count)
        ]
        self.encode_workers = [
            threading.Thread(
                target=self._encode_worker,
                args=(worker_idx,),
                name=f"concat-encode-{worker_idx}",
                daemon=True,
            )
            for worker_idx in range(self.worker_count)
        ]
        self.decode_workers = [
            threading.Thread(
                target=self._decode_worker,
                args=(worker_idx,),
                name=f"concat-decode-{worker_idx}",
                daemon=True,
            )
            for worker_idx in range(self.worker_count)
        ]
        self.transport_worker = threading.Thread(
            target=self._transport_worker,
            name="concat-transport",
            daemon=True,
        )
        for worker in self.encode_workers + self.decode_workers:
            worker.start()
        self.transport_worker.start()

    def _record_error(self, error: BaseException) -> None:
        with self.lock:
            if self.error is None:
                self.error = error

    def submit(self, task: PreRopeLayerTask) -> None:
        if task.layer_idx not in self.route_by_source:
            return
        while not self.inflight.acquire(timeout=0.1):
            if self.error is not None:
                raise RuntimeError("Concat layer pipeline failed.") from self.error
        if self.error is not None:
            self.inflight.release()
            raise RuntimeError("Concat layer pipeline failed.") from self.error
        self.encode_queue.put(task)

    @staticmethod
    def _stream_context(stream):
        return torch.cuda.stream(stream) if stream is not None else nullcontext()

    def _encode_worker(self, worker_idx: int) -> None:
        stream = self.encode_streams[worker_idx]
        try:
            while True:
                item = self.encode_queue.get()
                try:
                    if item is self._STOP:
                        return
                    task: PreRopeLayerTask = item
                    route = self.route_by_source[task.layer_idx]
                    original_bytes = int(
                        task.key.numel() * task.key.element_size()
                        + task.value.numel() * task.value.element_size()
                    )
                    with self._stream_context(stream):
                        if stream is not None and task.ready_event is not None:
                            stream.wait_event(task.ready_event)
                        started = time.perf_counter()
                        key_latent, value_latent = self.aligner.encode_layer(
                            route, task.key, task.value
                        )
                        if stream is not None:
                            stream.synchronize()
                        ended = time.perf_counter()
                        lcf_seconds = ended - started
                        lcf_interval = (started, ended)
                        latent_bytes = int(
                            key_latent.numel() * key_latent.element_size()
                            + value_latent.numel() * value_latent.element_size()
                        )
                        started = time.perf_counter()
                        payload = self.codec.encode_layer(
                            route[0],
                            self.num_layers,
                            key_latent,
                            value_latent,
                            self.codec_config,
                        )
                        if stream is not None:
                            stream.synchronize()
                        ended = time.perf_counter()
                        codec_seconds = ended - started
                        codec_interval = (started, ended)
                    payload_bytes = len(serialize_payload(payload))
                    with self.lock:
                        self.original_kv_bytes += original_bytes
                        self.latent_kv_bytes += latent_bytes
                        self.payload_bytes += payload_bytes
                        self.lcf_encode_seconds += lcf_seconds
                        self.codec_encode_seconds += codec_seconds
                        self.stage_intervals["lcf_encode"].append(lcf_interval)
                        self.stage_intervals["encode"].append(codec_interval)
                        self.layer_timings.setdefault(route[0], {}).update(
                            lcf_encode_seconds=lcf_seconds,
                            encode_seconds=codec_seconds,
                        )
                    self.transport_queue.put(_EncodedLayer(route, payload))
                except BaseException as error:
                    self._record_error(error)
                    self.inflight.release()
                finally:
                    self.encode_queue.task_done()
        finally:
            self.transport_queue.put(self._STOP)

    def _transport_worker(self) -> None:
        stopped = 0
        while stopped < self.worker_count:
            item = self.transport_queue.get()
            try:
                if item is self._STOP:
                    stopped += 1
                    continue
                encoded: _EncodedLayer = item
                try:
                    received = (
                        self.transport.roundtrip(encoded.payload)
                        if self.transport is not None
                        else encoded.payload
                    )
                    if self.transport is not None and self.transport.last_stats is not None:
                        with self.lock:
                            self.transport_stats.append(self.transport.last_stats)
                    self.decode_queue.put(_ReceivedLayer(encoded.route, received))
                except BaseException as error:
                    self._record_error(error)
                    self.inflight.release()
            finally:
                self.transport_queue.task_done()
        for _ in range(self.worker_count):
            self.decode_queue.put(self._STOP)

    def _decode_worker(self, worker_idx: int) -> None:
        stream = self.decode_streams[worker_idx]
        while True:
            item = self.decode_queue.get()
            try:
                if item is self._STOP:
                    return
                received: _ReceivedLayer = item
                try:
                    with self._stream_context(stream):
                        started = time.perf_counter()
                        decoded = self.codec.decode(received.payload, self.codec_config)
                        if stream is not None:
                            stream.synchronize()
                        ended = time.perf_counter()
                        codec_seconds = ended - started
                        codec_interval = (started, ended)
                        if len(decoded) != 1:
                            raise ValueError(
                                "A streamed concat payload must decode exactly one layer."
                            )
                        if self.zero_sharer_cache_at_receiver:
                            decoded = ((
                                torch.zeros_like(decoded[0][0]),
                                torch.zeros_like(decoded[0][1]),
                            ),)
                        started = time.perf_counter()
                        key, value = self.aligner.decode_layer(
                            received.route, decoded[0][0], decoded[0][1]
                        )
                        if stream is not None:
                            stream.synchronize()
                        ended = time.perf_counter()
                        lcf_seconds = ended - started
                        lcf_interval = (started, ended)
                    target_layer = received.route[0]
                    with self.lock:
                        if target_layer in self.decoded_by_layer:
                            raise ValueError(f"Duplicate Receiver layer {target_layer}.")
                        self.decoded_by_layer[target_layer] = (key, value)
                        self.codec_decode_seconds += codec_seconds
                        self.lcf_decode_seconds += lcf_seconds
                        self.stage_intervals["decode"].append(codec_interval)
                        self.stage_intervals["lcf_decode"].append(lcf_interval)
                        self.layer_timings.setdefault(target_layer, {}).update(
                            decode_seconds=codec_seconds,
                            lcf_decode_seconds=lcf_seconds,
                        )
                except BaseException as error:
                    self._record_error(error)
                finally:
                    self.inflight.release()
            finally:
                self.decode_queue.task_done()

    def finish(self):
        for _ in self.encode_workers:
            self.encode_queue.put(self._STOP)
        for worker in self.encode_workers:
            worker.join()
        self.transport_worker.join()
        for worker in self.decode_workers:
            worker.join()
        if self.error is not None:
            raise RuntimeError("Concat layer pipeline failed.") from self.error
        missing = sorted(set(range(self.num_layers)) - set(self.decoded_by_layer))
        if missing:
            raise ValueError(f"Concat pipeline is missing Receiver layers: {missing}.")
        sequence_length = int(next(iter(self.decoded_by_layer.values()))[0].shape[2])
        routing = LCFLatentRouting(
            routes=self.routing.routes,
            latent_dim=self.routing.latent_dim,
            sequence_length=sequence_length,
        )
        prefix = self.aligner.assemble_receiver_cache(self.decoded_by_layer, routing)
        return prefix, routing

    def abort(self) -> None:
        self._record_error(RuntimeError("Concat layer pipeline aborted."))
        self.finish()

    @property
    def pipeline_seconds(self) -> float:
        return float(time.perf_counter() - self.started_at)

    @staticmethod
    def _interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
        if not intervals:
            return 0.0
        ordered = sorted(intervals)
        total = 0.0
        start, end = ordered[0]
        for next_start, next_end in ordered[1:]:
            if next_start <= end:
                end = max(end, next_end)
            else:
                total += end - start
                start, end = next_start, next_end
        return float(total + end - start)

    def stage_wall_seconds(self, *stage_names: str) -> float:
        intervals = []
        for stage_name in stage_names:
            intervals.extend(self.stage_intervals[stage_name])
        return self._interval_union_seconds(intervals)

    def aggregate_transport_stats(self) -> TransportStats | None:
        if not self.transport_stats:
            return None
        return TransportStats(
            payload_bytes=sum(item.payload_bytes for item in self.transport_stats),
            serialize_seconds=sum(item.serialize_seconds for item in self.transport_stats),
            transmit_seconds=sum(item.transmit_seconds for item in self.transport_stats),
            deserialize_seconds=sum(item.deserialize_seconds for item in self.transport_stats),
        )
