#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Endpoint selection:
# - leave empty to use the environment/default
# - set to https://huggingface.co to force the official hub
# - set to https://hf-mirror.com to use the mirror
HF_ENDPOINT_OVERRIDE="${HF_ENDPOINT_OVERRIDE:-}"

# Whether to include the gated Llama teacher model.
# Set INCLUDE_GATED_MODELS=0 if you do not have access to Meta Llama.
INCLUDE_GATED_MODELS="${INCLUDE_GATED_MODELS:-1}"

# Download scope:
# - model    : model weights + tokenizer + config
# - minimal  : config/tokenizer files only
DOWNLOAD_SCOPE="${DOWNLOAD_SCOPE:-model}"

MODELS=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen2.5-Math-1.5B"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-4B-Base"
)


if [[ -n "${HF_ENDPOINT_OVERRIDE}" ]]; then
  export HF_ENDPOINT="${HF_ENDPOINT_OVERRIDE}"
fi

echo "Repo root: ${REPO_ROOT}"
echo "HF_ENDPOINT: ${HF_ENDPOINT:-<default>}"
echo "HF_HOME: ${HF_HOME:-<default>}"
echo "HF_HUB_CACHE: ${HF_HUB_CACHE:-<default>}"
echo "DOWNLOAD_SCOPE: ${DOWNLOAD_SCOPE}"
echo "INCLUDE_GATED_MODELS: ${INCLUDE_GATED_MODELS}"
echo
echo "Models to predownload:"
for model in "${MODELS[@]}"; do
  echo "  - ${model}"
done
echo

python - "${DOWNLOAD_SCOPE}" "${MODELS[@]}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

download_scope = sys.argv[1]
models = sys.argv[2:]

allow_patterns = None
ignore_patterns = None

if download_scope == "minimal":
    allow_patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "vocab.txt",
        "*.model",
    ]
elif download_scope == "model":
    # Download the full model snapshot.
    pass
else:
    raise ValueError(f"Unsupported DOWNLOAD_SCOPE: {download_scope}")

print("Resolved environment:")
print(f"  HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<default>')}")
print(f"  HF_HOME={os.environ.get('HF_HOME', '<default>')}")
print(f"  HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '<default>')}")
print()

for model_id in models:
    print(f"[START] {model_id}")
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
        print(f"[OK] {model_id}")
        print(f"     cached at: {local_dir}")
    except Exception as exc:
        print(f"[FAIL] {model_id}")
        print(f"       {type(exc).__name__}: {exc}")
    print()
PY

echo "Predownload script finished."
