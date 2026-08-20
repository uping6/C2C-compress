import torch

from rosetta.cachejpeg_rosetta.cache_aligner import LCFLatentRouting
from rosetta.cachejpeg_rosetta.concat_layer_streaming import ConcatLayerPipeline
from rosetta.cachejpeg_rosetta.pre_rope import (
    PreRopeLayerTask,
    StreamingPreRopeKVPublisher,
)


class _IdentityAligner:
    def encode_layer(self, _route, key, value):
        return key, value

    def decode_layer(self, _route, key, value):
        return key, value

    def assemble_receiver_cache(self, decoded_by_layer, _routing):
        return decoded_by_layer


class _IdentityLayerCodec:
    device = torch.device("cpu")

    def encode_layer(self, layer_idx, num_layers, key, value, _config):
        return {"layer_idx": layer_idx, "num_layers": num_layers, "kv": (key, value)}

    def decode(self, payload, _config):
        return (payload["kv"],)


def test_pre_rope_publisher_waits_for_both_components():
    published = []
    publisher = StreamingPreRopeKVPublisher(published.append)
    key = torch.ones(1, 1, 3, 4)
    value = torch.full_like(key, 2)

    publisher.capture_key(0, key)
    assert published == []
    publisher.capture_value(0, value)

    assert len(published) == 1
    assert published[0].layer_idx == 0
    assert torch.equal(published[0].key, key)
    assert torch.equal(published[0].value, value)


def test_concat_pipeline_routes_layers_and_collects_stats():
    routing = LCFLatentRouting(
        routes=((0, 1, 0), (1, 0, 0)), latent_dim=4, sequence_length=0
    )
    pipeline = ConcatLayerPipeline(
        aligner=_IdentityAligner(),
        codec=_IdentityLayerCodec(),
        codec_config={},
        transport=None,
        routing=routing,
        gpu_streams=2,
        max_inflight_layers=2,
    )
    for source_layer in (0, 1):
        tensor = torch.full((1, 1, 3, 4), float(source_layer + 1))
        pipeline.submit(PreRopeLayerTask(source_layer, tensor, tensor + 10))

    decoded, finished_routing = pipeline.finish()

    assert sorted(decoded) == [0, 1]
    assert torch.equal(decoded[0][0], torch.full((1, 1, 3, 4), 2.0))
    assert torch.equal(decoded[1][0], torch.full((1, 1, 3, 4), 1.0))
    assert finished_routing.sequence_length == 3
    assert pipeline.original_kv_bytes > 0
    assert pipeline.payload_bytes > 0
    assert sorted(pipeline.layer_timings) == [0, 1]

