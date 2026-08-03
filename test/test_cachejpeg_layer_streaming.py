import threading

import torch

from rosetta.cachejpeg_rosetta.layer_streaming import (
    LayerCompressionPipeline,
    LayerPrefillTimer,
    StreamingDynamicCache,
)


class DummyLayerCodec:
    device = torch.device("cpu")

    def __init__(self):
        self.encoded = []
        self.worker_threads = []

    def encode_layer(self, layer_idx, num_layers, key, value, config):
        self.encoded.append((layer_idx, num_layers))
        self.worker_threads.append(threading.current_thread().name)
        return {"layer_idx": layer_idx, "cache": (key + 1, value + 1)}

    def decode(self, payload, config):
        return (payload["cache"],)


class RecordingTransport:
    last_stats = None

    def __init__(self):
        self.layers = []

    def roundtrip(self, payload):
        self.layers.append(payload["layer_idx"])
        return payload


def test_dynamic_cache_publishes_layers_to_background_pipeline_in_order():
    codec = DummyLayerCodec()
    transport = RecordingTransport()
    pipeline = LayerCompressionPipeline(
        codec=codec,
        codec_config={},
        transport=transport,
        num_layers=3,
        queue_size=1,
    )
    cache = StreamingDynamicCache(pipeline.submit)

    for layer_idx in range(3):
        key = torch.full((1, 1, 4, 2), float(layer_idx))
        value = key + 10
        returned_key, _ = cache.update(key, value, layer_idx)
        assert torch.equal(returned_key, key)

    restored = pipeline.finish()

    assert codec.encoded == [(0, 3), (1, 3), (2, 3)]
    assert transport.layers == [0, 1, 2]
    assert all(name == "cachejpeg-layer-worker" for name in codec.worker_threads)
    assert [int(layer[0][0, 0, 0, 0]) for layer in restored] == [1, 2, 3]


def test_streaming_cache_only_publishes_first_update_per_layer():
    published = []
    cache = StreamingDynamicCache(lambda task: published.append(task.layer_idx))
    values = torch.zeros(1, 1, 2, 2)

    cache.update(values, values, 0)
    cache.update(values, values, 0)

    assert published == [0]
    assert cache.key_cache[0].shape[2] == 4


def test_layer_prefill_timer_records_each_decoder_layer():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])

        def forward(self, values):
            for layer in self.model.layers:
                values = layer(values)
            return values

    model = Model()
    timer = LayerPrefillTimer(model, 3)
    timer.start()
    model(torch.ones(1, 2))
    durations = timer.finish()

    assert len(durations) == 3
    assert all(value >= 0.0 for value in durations)
