"""Serial transport simulation for uncompressed Rosetta sharer KV caches."""

from __future__ import annotations

import pickle
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any


_LENGTH_HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class RawKVTransportStats:
    payload_bytes: int
    serialize_seconds: float
    transmit_seconds: float
    deserialize_seconds: float


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Rosetta raw-KV transport closed before the frame was complete.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class SerialSocketPairTransport:
    """A single, bandwidth-limited socket-pair transfer for one raw KV payload."""

    def __init__(
        self,
        *,
        bandwidth_bytes_per_sec: float,
        fixed_latency_ms: float = 0.0,
        timeout_seconds: float = 120.0,
        max_payload_bytes: int = 8 << 30,
    ) -> None:
        if bandwidth_bytes_per_sec <= 0:
            raise ValueError("rosetta transport.bandwidth_bytes_per_sec must be positive.")
        self.bandwidth_bytes_per_sec = float(bandwidth_bytes_per_sec)
        self.fixed_latency_ms = float(fixed_latency_ms)
        self.timeout_seconds = float(timeout_seconds)
        self.max_payload_bytes = int(max_payload_bytes)
        self.last_stats: RawKVTransportStats | None = None

    def roundtrip(self, payload: Any) -> Any:
        serialize_started = time.perf_counter()
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        serialize_seconds = time.perf_counter() - serialize_started
        if len(blob) > self.max_payload_bytes:
            raise ValueError(
                f"Rosetta raw-KV payload has {len(blob)} bytes, exceeding "
                f"max_payload_bytes={self.max_payload_bytes}."
            )

        sender, receiver = socket.socketpair()
        sender.settimeout(self.timeout_seconds)
        receiver.settimeout(self.timeout_seconds)
        send_errors: list[BaseException] = []

        def send() -> None:
            try:
                delay_seconds = (
                    max(0.0, self.fixed_latency_ms / 1000.0)
                    + len(blob) / self.bandwidth_bytes_per_sec
                )
                if delay_seconds:
                    time.sleep(delay_seconds)
                sender.sendall(_LENGTH_HEADER.pack(len(blob)))
                sender.sendall(blob)
                sender.shutdown(socket.SHUT_WR)
            except BaseException as exc:
                send_errors.append(exc)
            finally:
                sender.close()

        transmit_started = time.perf_counter()
        sender_thread = threading.Thread(target=send, name="rosetta-raw-kv-sender", daemon=True)
        sender_thread.start()
        try:
            payload_size = _LENGTH_HEADER.unpack(_recv_exact(receiver, _LENGTH_HEADER.size))[0]
            if payload_size > self.max_payload_bytes:
                raise ValueError(
                    f"Rosetta raw-KV transport frame has {payload_size} bytes, exceeding "
                    f"max_payload_bytes={self.max_payload_bytes}."
                )
            received_blob = _recv_exact(receiver, payload_size)
        finally:
            receiver.close()
            sender_thread.join(timeout=self.timeout_seconds)
        if sender_thread.is_alive():
            raise TimeoutError("Rosetta raw-KV sender did not finish before the timeout.")
        if send_errors:
            raise send_errors[0]
        transmit_seconds = time.perf_counter() - transmit_started

        deserialize_started = time.perf_counter()
        restored = pickle.loads(received_blob)
        deserialize_seconds = time.perf_counter() - deserialize_started
        self.last_stats = RawKVTransportStats(
            payload_bytes=len(blob),
            serialize_seconds=serialize_seconds,
            transmit_seconds=transmit_seconds,
            deserialize_seconds=deserialize_seconds,
        )
        return restored


def build_rosetta_transport(config: dict[str, Any] | None) -> SerialSocketPairTransport | None:
    """Build the uncompressed C2C transport, rejecting unsupported parallel modes."""
    cfg = dict(config or {})
    mode = str(cfg.get("mode", "none")).lower()
    if mode in {"none", "off", "disabled"}:
        return None
    if mode not in {"socketpair", "socket", "unix"}:
        raise ValueError("rosetta transport.mode must be 'socketpair' or 'none'.")
    if bool(cfg.get("parallel", False)):
        raise ValueError("Pure Rosetta transport currently supports only parallel=false.")
    if cfg.get("bandwidth_bytes_per_sec") is None:
        raise ValueError("rosetta transport requires bandwidth_bytes_per_sec.")
    return SerialSocketPairTransport(
        bandwidth_bytes_per_sec=float(cfg["bandwidth_bytes_per_sec"]),
        fixed_latency_ms=float(cfg.get("fixed_latency_ms", 0.0)),
        timeout_seconds=float(cfg.get("timeout_seconds", 120.0)),
        max_payload_bytes=int(cfg.get("max_payload_bytes", 8 << 30)),
    )
