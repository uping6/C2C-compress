# CacheJPEG Eval Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `cachejpeg` evaluation method to `C2C-compress-master` that reuses the existing unified evaluation workflow while preserving all current HF and Rosetta behavior.

**Architecture:** Introduce a small CacheJPEG adapter layer that wraps a standard Hugging Face causal LM, performs prompt prefill, compresses/decompresses KV cache with a local codec implementation, and then resumes generation from the reconstructed cache. Keep `unified_evaluator.py` as the orchestration entry point and branch only on a new explicit model type/config section. Avoid modifying existing Rosetta internals.

**Tech Stack:** Python, PyTorch, Transformers, pytest, existing `unified_evaluator.py` pipeline, adapted CacheJPEG codec logic from `HomoC2C-KV`.

---

### Task 1: Add failing tests for config detection and evaluator dispatch

**Files:**
- Create: `test/test_cachejpeg_integration.py`
- Test: `test/test_cachejpeg_integration.py`

- [ ] **Step 1: Write the failing test**

```python
from script.evaluation.unified_evaluator import UnifiedEvaluator


def _base_config(model_name="cachejpeg"):
    return {
        "model": {
            "model_name": model_name,
            "base_model_name": "Qwen/Qwen3-0.6B",
            "generation_config": {"max_new_tokens": 8, "do_sample": False},
        },
        "output": {"output_dir": "local/test_outputs/cachejpeg"},
        "eval": {
            "dataset": "longbench",
            "gpu_ids": [0],
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
        },
    }


def test_cachejpeg_model_type_detection():
    evaluator = UnifiedEvaluator(_base_config())
    assert evaluator.is_cachejpeg_model is True


def test_non_cachejpeg_model_type_detection():
    evaluator = UnifiedEvaluator(_base_config(model_name="Rosetta"))
    assert evaluator.is_cachejpeg_model is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q test/test_cachejpeg_integration.py -k detection`
Expected: FAIL because `UnifiedEvaluator` does not expose `is_cachejpeg_model`.

- [ ] **Step 3: Write minimal implementation**

```python
# inside UnifiedEvaluator.__init__
self.is_cachejpeg_model = self.model_config["model_name"].lower() == "cachejpeg"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q test/test_cachejpeg_integration.py -k detection`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_cachejpeg_integration.py script/evaluation/unified_evaluator.py
git commit -m "test: add cachejpeg evaluator detection coverage"
```

### Task 2: Add failing tests for CacheJPEG config normalization

**Files:**
- Create: `test/test_cachejpeg_config.py`
- Create: `rosetta/cachejpeg/config.py`
- Test: `test/test_cachejpeg_config.py`

- [ ] **Step 1: Write the failing test**

```python
from rosetta.cachejpeg.config import resolve_cachejpeg_eval_config


def test_resolve_cachejpeg_eval_config_reads_nested_values():
    cfg = resolve_cachejpeg_eval_config(
        {
            "method": "cachejpeg",
            "anchors": {"sink_count": 2},
            "block": {"mode": "global", "size": 64},
            "quant": {"low": 1.0, "high": 8.0, "curve": "quadratic"},
            "entropy": {"representation": "dense_int16", "backend": "zlib1"},
        }
    )
    assert cfg.method == "cachejpeg"
    assert cfg.anchors.sink_count == 2
    assert cfg.block.mode == "global"
    assert cfg.entropy.backend == "zlib1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q test/test_cachejpeg_config.py`
Expected: FAIL with import error because `rosetta.cachejpeg.config` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnchorConfig:
    sink_count: int = 1


@dataclass(frozen=True)
class BlockConfig:
    mode: str = "global"
    size: int = 64


@dataclass(frozen=True)
class QuantConfig:
    low: float = 1.0
    high: float = 8.0
    curve: str = "quadratic"


@dataclass(frozen=True)
class EntropyConfig:
    representation: str = "dense_int16"
    backend: str = "zlib1"


@dataclass(frozen=True)
class CacheJPEGEvalConfig:
    method: str = "cachejpeg"
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)


def resolve_cachejpeg_eval_config(config: dict) -> CacheJPEGEvalConfig:
    return CacheJPEGEvalConfig(
        method=str(config.get("method", "cachejpeg")),
        anchors=AnchorConfig(sink_count=int((config.get("anchors") or {}).get("sink_count", 1))),
        block=BlockConfig(**(config.get("block") or {})),
        quant=QuantConfig(**(config.get("quant") or {})),
        entropy=EntropyConfig(**(config.get("entropy") or {})),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q test/test_cachejpeg_config.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_cachejpeg_config.py rosetta/cachejpeg/config.py
git commit -m "feat: add cachejpeg evaluation config parsing"
```

### Task 3: Add failing tests for CacheJPEG wrapper generation path

**Files:**
- Create: `test/test_cachejpeg_wrapper.py`
- Create: `rosetta/cachejpeg/wrapper.py`
- Test: `test/test_cachejpeg_wrapper.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from rosetta.cachejpeg.wrapper import CacheJPEGEvalWrapper


class DummyTokenizer:
    eos_token_id = 99

    def __call__(self, text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "decoded"


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True):
        logits = torch.zeros((1, input_ids.shape[1], 8))
        return type("Out", (), {"logits": logits, "past_key_values": ()})()


def test_cachejpeg_wrapper_generate_returns_tensor():
    wrapper = CacheJPEGEvalWrapper(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        codec_config={"method": "cachejpeg"},
    )
    output = wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(output, torch.Tensor)
    assert output.ndim == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q test/test_cachejpeg_wrapper.py`
Expected: FAIL because `CacheJPEGEvalWrapper` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
class CacheJPEGEvalWrapper:
    def __init__(self, model, tokenizer, codec_config):
        self.model = model
        self.tokenizer = tokenizer
        self.codec_config = codec_config

    def generate(self, input_ids, attention_mask=None, **generation_config):
        return input_ids[:, -1:].clone()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q test/test_cachejpeg_wrapper.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_cachejpeg_wrapper.py rosetta/cachejpeg/wrapper.py
git commit -m "test: add cachejpeg wrapper generation scaffold"
```

### Task 4: Implement the real CacheJPEG wrapper using adapted KV-cache compression

**Files:**
- Create: `rosetta/cachejpeg/__init__.py`
- Modify: `rosetta/cachejpeg/config.py`
- Modify: `rosetta/cachejpeg/wrapper.py`
- Create: `rosetta/cachejpeg/cache_utils.py`
- Create: `rosetta/cachejpeg/codec.py`
- Test: `test/test_cachejpeg_wrapper.py`

- [ ] **Step 1: Extend the failing test for cache roundtrip hook usage**

```python
def test_cachejpeg_wrapper_uses_codec_roundtrip(monkeypatch):
    calls = []

    class DummyCodec:
        def encode(self, past_key_values, config):
            calls.append("encode")
            return {"payload": past_key_values}

        def decode(self, payload, config):
            calls.append("decode")
            return payload["payload"]

    wrapper = CacheJPEGEvalWrapper(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        codec_config={"method": "cachejpeg"},
        codec=DummyCodec(),
    )
    wrapper.generate(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        max_new_tokens=1,
        do_sample=False,
    )
    assert calls == ["encode", "decode"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q test/test_cachejpeg_wrapper.py -k roundtrip`
Expected: FAIL because wrapper does not invoke codec.

- [ ] **Step 3: Write minimal implementation**

```python
# inside CacheJPEGEvalWrapper.generate
prefill_outputs = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=self._build_position_ids(attention_mask),
    use_cache=True,
)
payload = self.codec.encode(prefill_outputs.past_key_values, self.codec_config)
restored = self.codec.decode(payload, self.codec_config)
return self._decode_from_cache(
    last_token=input_ids[:, -1:],
    past_key_values=restored,
    max_new_tokens=generation_config.get("max_new_tokens", 1),
    do_sample=generation_config.get("do_sample", False),
    temperature=generation_config.get("temperature", 0.0),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q test/test_cachejpeg_wrapper.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add rosetta/cachejpeg/__init__.py rosetta/cachejpeg/config.py rosetta/cachejpeg/wrapper.py rosetta/cachejpeg/cache_utils.py rosetta/cachejpeg/codec.py test/test_cachejpeg_wrapper.py
git commit -m "feat: add cachejpeg kv compression wrapper"
```

### Task 5: Wire CacheJPEG into `unified_evaluator.py` and add config coverage

**Files:**
- Modify: `script/evaluation/unified_evaluator.py`
- Modify: `recipe/eval_recipe/unified_eval.yaml`
- Modify: `recipe/eval_recipe/longbench_eval.yaml`
- Test: `test/test_cachejpeg_integration.py`

- [ ] **Step 1: Extend failing test for loader dispatch**

```python
from unittest.mock import patch

from script.evaluation.unified_evaluator import UnifiedEvaluator


def test_cachejpeg_model_dispatch_uses_cachejpeg_loader():
    evaluator = UnifiedEvaluator(_base_config())
    with patch("script.evaluation.unified_evaluator.load_cachejpeg_model") as mocked_loader:
        mocked_loader.return_value = object(), object()
        with patch.object(evaluator, "evaluate_subject", return_value=([], 0.0, None, [], [])):
            evaluator.evaluate_on_gpu(rank=0, gpu_id=0, subjects=["qasper"], return_dict={})
    assert mocked_loader.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q test/test_cachejpeg_integration.py -k dispatch`
Expected: FAIL because `load_cachejpeg_model` does not exist or is never called.

- [ ] **Step 3: Write minimal implementation**

```python
# unified_evaluator.py
from rosetta.cachejpeg.wrapper import load_cachejpeg_model

...
elif self.is_cachejpeg_model:
    model, tokenizer = load_cachejpeg_model(self.model_config, device=device, generation_config=self.generation_config)
    model_type = "cachejpeg"
    llm_tokenizer = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q test/test_cachejpeg_integration.py -k dispatch`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add script/evaluation/unified_evaluator.py recipe/eval_recipe/unified_eval.yaml recipe/eval_recipe/longbench_eval.yaml test/test_cachejpeg_integration.py
git commit -m "feat: wire cachejpeg into unified evaluator"
```

### Task 6: Verify regression safety for existing evaluator paths

**Files:**
- Test: `test/test_cachejpeg_integration.py`

- [ ] **Step 1: Add failing regression tests for Rosetta/HF detection**

```python
def test_rosetta_detection_still_works():
    evaluator = UnifiedEvaluator(_base_config(model_name="Rosetta"))
    assert evaluator.use_two_stage is False
    assert evaluator.is_cachejpeg_model is False


def test_hf_detection_still_works():
    evaluator = UnifiedEvaluator(_base_config(model_name="Qwen/Qwen3-0.6B"))
    assert evaluator.is_cachejpeg_model is False
```

- [ ] **Step 2: Run test to verify expected status**

Run: `pytest -q test/test_cachejpeg_integration.py`
Expected: PASS after earlier implementation.

- [ ] **Step 3: Refactor only if needed**

```python
# keep evaluator detection explicit and ordered:
self.model_name_normalized = self.model_config["model_name"].lower()
self.is_cachejpeg_model = self.model_name_normalized == "cachejpeg"
self.use_two_stage = self.model_name_normalized in ["two_stage", "two_stage_rosetta"]
```

- [ ] **Step 4: Run focused tests and then full new suite**

Run: `pytest -q test/test_cachejpeg_config.py test/test_cachejpeg_wrapper.py test/test_cachejpeg_integration.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add test/test_cachejpeg_integration.py script/evaluation/unified_evaluator.py
git commit -m "test: lock cachejpeg integration against evaluator regressions"
```
