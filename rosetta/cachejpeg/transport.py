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
class TransportStats:
    payload_bytes: int
    serialize_seconds: float
    transmit_seconds: float
    deserialize_seconds: float


def serialize_payload(payload: Any) -> bytes:
    """Match HomoC2C-KV's pickle wire representation."""
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_payload(blob: bytes) -> Any:
    return pickle.loads(blob)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("CacheJPEG transport closed before the frame was complete.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, blob: bytes) -> None:
    sock.sendall(_LENGTH_HEADER.pack(len(blob)))
    sock.sendall(blob)


def _receive_frame(sock: socket.socket, max_payload_bytes: int) -> bytes:
    payload_size = _LENGTH_HEADER.unpack(_recv_exact(sock, _LENGTH_HEADER.size))[0]
    if payload_size > max_payload_bytes:
        raise ValueError(
            f"CacheJPEG transport frame has {payload_size} bytes, exceeding "
            f"max_payload_bytes={max_payload_bytes}."
        )
    return _recv_exact(sock, payload_size)


class SocketPairTransport:
    """Transfer a payload through a real framed kernel socket.

    The evaluator currently hosts sharer and receiver in one process, so a
    Unix socket pair provides real byte transport without a second service.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        max_payload_bytes: int = 8 << 30,
        bandwidth_bytes_per_sec: float | None = None,
        fixed_latency_ms: float = 0.0,
    ):
        self.timeout_seconds = float(timeout_seconds)
        self.max_payload_bytes = int(max_payload_bytes)
        self.bandwidth_bytes_per_sec = (
            float(bandwidth_bytes_per_sec) if bandwidth_bytes_per_sec else None
        )
        self.fixed_latency_ms = float(fixed_latency_ms)
        self.last_stats: TransportStats | None = None

    def roundtrip(self, payload: Any) -> Any:
        started = time.perf_counter()
        blob = serialize_payload(payload)
        serialize_seconds = time.perf_counter() - started

        sender, receiver = socket.socketpair()
        sender.settimeout(self.timeout_seconds)
        receiver.settimeout(self.timeout_seconds)
        send_error: list[BaseException] = []

        def send() -> None:
            try:
                delay_seconds = max(0.0, self.fixed_latency_ms / 1000.0)
                if self.bandwidth_bytes_per_sec is not None:
                    if self.bandwidth_bytes_per_sec <= 0:
                        raise ValueError("transport.bandwidth_bytes_per_sec must be positive.")
                    delay_seconds += len(blob) / self.bandwidth_bytes_per_sec
                if delay_seconds:
                    time.sleep(delay_seconds)
                _send_frame(sender, blob)
                sender.shutdown(socket.SHUT_WR)
            except BaseException as exc:
                send_error.append(exc)
            finally:
                sender.close()

        transmit_started = time.perf_counter()
        sender_thread = threading.Thread(target=send, name="cachejpeg-sender", daemon=True)
        sender_thread.start()
        receive_error: BaseException | None = None
        received_blob = b""
        try:
            received_blob = _receive_frame(receiver, self.max_payload_bytes)
        except BaseException as exc:
            receive_error = exc
        finally:
            receiver.close()
            sender_thread.join(timeout=self.timeout_seconds)
        if sender_thread.is_alive():
            raise TimeoutError("CacheJPEG sender did not finish before the transport timeout.")
        if receive_error is not None:
            raise receive_error
        if send_error:
            raise send_error[0]
        transmit_seconds = time.perf_counter() - transmit_started

        deserialize_started = time.perf_counter()
        restored = deserialize_payload(received_blob)
        deserialize_seconds = time.perf_counter() - deserialize_started
        self.last_stats = TransportStats(
            payload_bytes=len(blob),
            serialize_seconds=serialize_seconds,
            transmit_seconds=transmit_seconds,
            deserialize_seconds=deserialize_seconds,
        )
        return restored


class DirectTransport:
    """Compatibility mode that still crosses a serialization boundary."""

    def __init__(self):
        self.last_stats: TransportStats | None = None

    def roundtrip(self, payload: Any) -> Any:
        started = time.perf_counter()
        blob = serialize_payload(payload)
        serialize_seconds = time.perf_counter() - started
        deserialize_started = time.perf_counter()
        restored = deserialize_payload(blob)
        deserialize_seconds = time.perf_counter() - deserialize_started
        self.last_stats = TransportStats(
            payload_bytes=len(blob),
            serialize_seconds=serialize_seconds,
            transmit_seconds=0.0,
            deserialize_seconds=deserialize_seconds,
        )
        return restored


def build_transport(config: dict[str, Any] | None):
    cfg = config or {}
    mode = str(cfg.get("mode", "none")).lower()
    if mode in {"socketpair", "socket", "unix"}:
        return SocketPairTransport(
            timeout_seconds=float(cfg.get("timeout_seconds", 120.0)),
            max_payload_bytes=int(cfg.get("max_payload_bytes", 8 << 30)),
            bandwidth_bytes_per_sec=(
                float(cfg["bandwidth_bytes_per_sec"])
                if cfg.get("bandwidth_bytes_per_sec") is not None
                else None
            ),
            fixed_latency_ms=float(cfg.get("fixed_latency_ms", 0.0)),
        )
    if mode in {"direct", "serialize_only"}:
        return DirectTransport()
    if mode in {"none", "off", "disabled"}:
        return None
    raise ValueError(f"Unsupported cachejpeg.transport.mode: {mode}")
