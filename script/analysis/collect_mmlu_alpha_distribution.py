"""Collect inference-time adaptive-quantizer alpha selections on local MMLU-Redux."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from rosetta.utils.evaluate import build_prompt
from script.train.SFT_train import setup_models


def chat_prompt(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    model_cfg = cfg["model"]
    model, base_tokenizer, _aligner, teacher_tokenizer = setup_models(
        model_cfg, "rosetta", device=args.device, dtype=torch.bfloat16
    )
    model.to(args.device).eval()
    checkpoint = Path(args.checkpoint)
    for index, projector in enumerate(model.projector_list):
        projector.load_state_dict(torch.load(checkpoint / f"projector_{index}.pt", map_location="cpu"))
    model.adaptive_quant_table.load_state_dict(
        torch.load(checkpoint / "adaptive_quant_table.pt", map_location="cpu")
    )
    model.adaptive_quant_table.eval()

    counts = Counter()
    per_layer = defaultdict(Counter)
    per_subject = defaultdict(Counter)
    samples = 0
    with Path(args.data).open(encoding="utf-8") as handle, torch.inference_mode():
        for line in handle:
            row = json.loads(line)
            choices = "\n".join(
                f"{chr(65 + i)}. {choice}" for i, choice in enumerate(row["endings"])
            )
            prompt = build_prompt("mmlu-redux", "en", row["ctx"], choices, use_cot=False)
            base = base_tokenizer(chat_prompt(base_tokenizer, prompt), return_tensors="pt").to(args.device)
            teacher = teacher_tokenizer(chat_prompt(teacher_tokenizer, prompt), return_tensors="pt").to(args.device)
            # Concat LCF uses the first non--100 label only to delimit the
            # receiver prompt.  Marking its last token supervised retains the
            # complete prompt while avoiding answer generation for this study.
            labels = torch.full_like(base.input_ids, -100)
            labels[:, -1] = base.input_ids[:, -1]
            model(
                input_ids=[base.input_ids, teacher.input_ids],
                attention_mask=[base.attention_mask, teacher.attention_mask],
                labels=labels,
                sharer_prompt_only=True,
                use_cache=True,
            )
            indices = model.adaptive_quant_table.last_result.table_indices[0].detach().cpu()
            for layer, kv, head in torch.nonzero(torch.ones_like(indices), as_tuple=False).tolist():
                choice = int(indices[layer, kv, head])
                counts[choice] += 1
                per_layer[str(layer)][choice] += 1
                per_subject[row.get("subject", "unknown")][choice] += 1
            samples += 1
            if samples % 50 == 0:
                print(f"processed={samples}", flush=True)

    candidates = model.adaptive_quant_table.alpha_candidates.detach().cpu().tolist()
    total = sum(counts.values())
    result = {
        "samples": samples,
        "groups_per_sample": int(indices.numel()),
        "total_selections": total,
        "alpha_candidates": candidates,
        "counts": {str(candidates[i]): counts[i] for i in range(len(candidates))},
        "proportions": {str(candidates[i]): counts[i] / total for i in range(len(candidates))},
        "mean_alpha": sum(candidates[i] * counts[i] for i in range(len(candidates))) / total,
        "per_receiver_layer_counts": {
            key: {str(candidates[i]): value[i] for i in range(len(candidates))}
            for key, value in per_layer.items()
        },
        "per_subject_counts": {
            key: {str(candidates[i]): value[i] for i in range(len(candidates))}
            for key, value in per_subject.items()
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
