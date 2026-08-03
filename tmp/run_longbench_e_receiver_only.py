#!/usr/bin/env python3
"""Run the LongBench-E receiver-only ablation for the CacheJPEG-Rosetta setup."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml


ROOT = Path("/data/smy_data")
BASE_CONFIG = ROOT / "recipe/eval_recipe/longbench_jpegcache.yaml"
DEFAULT_OUTPUT = ROOT / "tmp/longbench_e_receiver_only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="Evaluate rows [0, 4) for a quick check.")
    args = parser.parse_args()

    config = deepcopy(yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8")))
    config["output"]["output_dir"] = str(args.output_dir)
    config["eval"]["longbench_e"] = True
    config["eval"]["skip_existing_longbench_subjects"] = False
    config["model"]["cachejpeg_rosetta_config"]["ablation"] = {
        "receiver_only": True,
    }
    if args.smoke:
        config["eval"]["limit"] = [0, 4]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = args.output_dir / "experiment_config.yaml"
    experiment_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "script/evaluation/unified_evaluator.py", "--config", str(experiment_config)],
        cwd=ROOT,
        check=True,
    )

    summaries = sorted(args.output_dir.glob("*_summary.json"), key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise RuntimeError(f"No evaluation summary was written under {args.output_dir}")
    summary_path = summaries[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "experiment": "longbench_e_receiver_only",
                "summary": str(summary_path),
                "num_subjects": len(summary.get("subjects", {})),
                "final_score": summary.get("final_score", summary.get("overall_accuracy")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
