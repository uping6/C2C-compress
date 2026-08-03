import pytest
import time

from rosetta.cachejpeg.transport import SocketPairTransport, build_transport


def test_socketpair_transport_moves_a_framed_serialized_payload():
    payload = {"method": "cachejpeg", "data": b"abc" * 100_000, "layers": [0, 1, 2]}
    transport = SocketPairTransport(timeout_seconds=5)

    restored = transport.roundtrip(payload)

    assert restored == payload
    assert restored is not payload
    assert transport.last_stats is not None
    assert transport.last_stats.payload_bytes > len(payload["data"])
    assert transport.last_stats.transmit_seconds >= 0.0


def test_transport_rejects_frames_above_receiver_limit():
    transport = SocketPairTransport(timeout_seconds=5, max_payload_bytes=8)
    with pytest.raises(ValueError, match="max_payload_bytes"):
        transport.roundtrip({"payload": b"too large"})


def test_build_transport_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported cachejpeg.transport.mode"):
        build_transport({"mode": "unknown"})


def test_socketpair_transport_applies_bandwidth_limit():
    transport = SocketPairTransport(
        timeout_seconds=5,
        bandwidth_bytes_per_sec=100_000,
    )
    started = time.perf_counter()
    transport.roundtrip({"data": b"x" * 20_000})
    elapsed = time.perf_counter() - started

    assert elapsed >= 0.18


def test_build_transport_none_disables_transport():
    assert build_transport({"mode": "none"}) is None


def test_build_transport_defaults_to_disabled_for_backward_compatibility():
    assert build_transport(None) is None
    assert build_transport({}) is None
