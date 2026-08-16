"""
Unified Evaluation Script for Multiple Benchmarks

This script provides a unified interface for evaluating models on various benchmarks
including MMLU-Redux, MMMLU, and LongBench. It supports multi-GPU parallel evaluation and 
multiple answer generation methods.

Usage:
    python unified_evaluator.py --config eval_recipe/config.yaml
"""

import argparse
import os

import json
import yaml
import csv
import torch
import torch.multiprocessing as mp
from collections import defaultdict
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from datasets import load_dataset, load_from_disk
from datetime import datetime
import hashlib
import random
import time
import re
import sys
import re
import hashlib
from collections import Counter

from rosetta.utils.evaluate import (
    extract_answer_from_content,
    load_hf_model,
    load_rosetta_model,
    get_option_token_ids,
    build_prompt,
    apply_generation_config
)
from rosetta.utils.matheval import GSM8KEvaluator, MATH500Evaluator
from rosetta.model.wrapper import RosettaModel
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from transformers import AutoTokenizer
from rosetta.utils.evaluate import set_default_chat_template
from rosetta.baseline.multi_stage import TwoStageInference, TwoStageRosetta
from rosetta.cachejpeg.wrapper import load_cachejpeg_model
from rosetta.cachejpeg_rosetta.wrapper import load_cachejpeg_rosetta_model

# Dataset-specific configurations
DATASET_CONFIGS = {
    "mmlu-redux": {
        "dataset_name": "edinburgh-dawg/mmlu-redux-2.0",
        "test_split": "test",
        "subjects": [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
            'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
            'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
            'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
            'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
            'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history',
            'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence',
            'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics',
            'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory',
            'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology',
            'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions'
        ],
        "subcategories": {
            "abstract_algebra": ["math"],
            "anatomy": ["health"],
            "astronomy": ["physics"],
            "business_ethics": ["business"],
            "clinical_knowledge": ["health"],
            "college_biology": ["biology"],
            "college_chemistry": ["chemistry"],
            "college_computer_science": ["computer science"],
            "college_mathematics": ["math"],
            "college_medicine": ["health"],
            "college_physics": ["physics"],
            "computer_security": ["computer science"],
            "conceptual_physics": ["physics"],
            "econometrics": ["economics"],
            "electrical_engineering": ["engineering"],
            "elementary_mathematics": ["math"],
            "formal_logic": ["philosophy"],
            "global_facts": ["other"],
            "high_school_biology": ["biology"],
            "high_school_chemistry": ["chemistry"],
            "high_school_computer_science": ["computer science"],
            "high_school_european_history": ["history"],
            "high_school_geography": ["geography"],
            "high_school_government_and_politics": ["politics"],
            "high_school_macroeconomics": ["economics"],
            "high_school_mathematics": ["math"],
            "high_school_microeconomics": ["economics"],
            "high_school_physics": ["physics"],
            "high_school_psychology": ["psychology"],
            "high_school_statistics": ["math"],
            "high_school_us_history": ["history"],
            "high_school_world_history": ["history"],
            "human_aging": ["health"],
            "human_sexuality": ["culture"],
            "international_law": ["law"],
            "jurisprudence": ["law"],
            "logical_fallacies": ["philosophy"],
            "machine_learning": ["computer science"],
            "management": ["business"],
            "marketing": ["business"],
            "medical_genetics": ["health"],
            "miscellaneous": ["other"],
            "moral_disputes": ["philosophy"],
            "moral_scenarios": ["philosophy"],
            "nutrition": ["health"],
            "philosophy": ["philosophy"],
            "prehistory": ["history"],
            "professional_accounting": ["other"],
            "professional_law": ["law"],
            "professional_medicine": ["health"],
            "professional_psychology": ["psychology"],
            "public_relations": ["politics"],
            "security_studies": ["politics"],
            "sociology": ["culture"],
            "us_foreign_policy": ["politics"],
            "virology": ["health"],
            "world_religions": ["philosophy"]
        },
        "categories": {
            "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
            "humanities": ["history", "philosophy", "law"],
            "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
            "other (business, health, misc.)": ["other", "business", "health"]
        }
    },
    "mmmlu": {
        "dataset_name": "openai/MMMLU",
        "test_split": "test",
        "subjects": [
            'AR_XY', 'BN_BD', 'DE_DE', 'ES_LA', 'FR_FR', 'HI_IN', 'ID_ID',
            'IT_IT', 'JA_JP', 'KO_KR', 'PT_BR', 'SW_KE', 'YO_NG', 'ZH_CN'
        ],
        "subcategories": {},  # MMMLU doesn't have subcategories
        "categories": {}  # MMMLU doesn't have categories
    },

    "gpqa": {
        "dataset_name": "Idavidrein/gpqa",
        "test_split": "train",
        "subjects": [
            "gpqa_diamond",
        ],
        "subcategories": {},
        "categories": {}
    },
    "math-500": {
        "dataset_name": "HuggingFaceH4/MATH-500",
        "test_split": "test",
        "subjects": ["all"]
    },
    "longbench": {
        "dataset_name": "THUDM/LongBench",
        "test_split": "test",
        "subjects": [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique",
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht",
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ],
        "subjects_e": ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
        "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"],
        "subcategories": {},  # LongBench doesn't have subcategories
        "categories": {}  # LongBench doesn't have categories
    },
    "gsm8k": {
        "dataset_name": "openai/gsm8k",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "openbookqa": {
        "dataset_name": "openbookqa",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "ai2-arc": {
        "dataset_name": "allenai/ai2_arc",
        "test_split": "test",
        "subjects": ["ARC-Challenge"],
        "subcategories": {},
        "categories": {}
    },
    "mmlu-pro": {
        "dataset_name": "TIGER-Lab/MMLU-Pro",
        "test_split": "test",
        "subjects": ["main"],
        "subcategories": {},
        "categories": {}
    },
    "ceval": {
        "dataset_name": "ceval/ceval-exam",
        "test_split": "test",
        "subjects": [
            "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
            "business_administration", "chinese_language_and_literature", "civil_servant",
            "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
            "college_programming", "computer_architecture", "computer_network",
            "discrete_mathematics", "education_science", "electrical_engineer",
            "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
            "high_school_chemistry", "high_school_chinese", "high_school_geography",
            "high_school_history", "high_school_mathematics", "high_school_physics",
            "high_school_politics", "ideological_and_moral_cultivation", "law",
            "legal_professional", "logic", "mao_zedong_thought", "marxism",
            "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
            "middle_school_geography", "middle_school_history", "middle_school_mathematics",
            "middle_school_physics", "middle_school_politics", "modern_chinese_history",
            "operating_system", "physician", "plant_protection", "probability_and_statistics",
            "professional_tour_guide", "sports_science", "tax_accountant",
            "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine"
        ],
        "subcategories": {},
        "categories": {}
    }
}


class UnifiedEvaluator:
    """Unified evaluator for multiple benchmark datasets."""

    @staticmethod
    def _resolve_model_ref(model_ref: Any, local_path_keys: tuple[str, ...], model_cfg: Dict[str, Any]) -> Any:
        """Prefer an existing local path; otherwise keep the HF model id."""
        if isinstance(model_ref, dict):
            return {
                key: UnifiedEvaluator._resolve_model_ref(value, local_path_keys, model_cfg)
                for key, value in model_ref.items()
            }
        if not isinstance(model_ref, str):
            return model_ref

        candidates: list[str] = []
        for key in local_path_keys:
            candidate = model_cfg.get(key)
            if candidate:
                candidates.append(str(candidate))
        candidates.append(model_ref)

        for candidate in candidates:
            candidate_path = Path(str(candidate)).expanduser()
            if candidate_path.exists():
                return str(candidate_path)
        return model_ref

    @staticmethod
    def _build_sharer_shuffle_pairs(
        sample_indices: List[int],
        *,
        seed: int,
        subject: str,
    ) -> Dict[int, int]:
        """Build a deterministic no-self-match permutation for one subject."""
        if len(sample_indices) < 2:
            raise ValueError(
                "shuffle_sharer_cache requires at least two evaluated samples "
                f"for subject {subject!r}; got {len(sample_indices)}"
            )
        stable_subject_seed = int(
            hashlib.sha256(str(subject).encode("utf-8")).hexdigest()[:16], 16
        )
        rng = random.Random(int(seed) ^ stable_subject_seed)
        offset = rng.randrange(1, len(sample_indices))
        shuffled = sample_indices[offset:] + sample_indices[:offset]
        pairs = dict(zip(sample_indices, shuffled))
        if any(receiver_idx == sharer_idx for receiver_idx, sharer_idx in pairs.items()):
            raise AssertionError("Internal error: sharer shuffle produced a self-match")
        return pairs

    @staticmethod
    def _compose_shuffled_sharer_inputs(
        receiver_inputs: Dict[str, Any],
        sharer_inputs: Dict[str, Any],
        *,
        teacher_pad_token_id: int,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Keep receiver sample i unchanged and replace only its teacher stream with
        sample j. The teacher stream is resized to the receiver cache length,
        which is required by the position-wise Rosetta projector.
        """
        receiver_ids_value = receiver_inputs["input_ids"]
        receiver_mask_value = receiver_inputs.get("attention_mask")
        sharer_ids_value = sharer_inputs["input_ids"]
        sharer_mask_value = sharer_inputs.get("attention_mask")

        receiver_ids = (
            receiver_ids_value[0]
            if isinstance(receiver_ids_value, (list, tuple))
            else receiver_ids_value
        )
        receiver_mask = (
            receiver_mask_value[0]
            if isinstance(receiver_mask_value, (list, tuple))
            else receiver_mask_value
        )
        teacher_ids = (
            sharer_ids_value[1]
            if isinstance(sharer_ids_value, (list, tuple))
            else sharer_ids_value
        )
        teacher_mask = (
            sharer_mask_value[1]
            if isinstance(sharer_mask_value, (list, tuple))
            else sharer_mask_value
        )

        if receiver_ids.ndim != 2 or teacher_ids.ndim != 2:
            raise ValueError("Shuffled receiver/sharer input_ids must be rank-2 tensors")
        if receiver_ids.shape[0] != 1 or teacher_ids.shape[0] != 1:
            raise ValueError("shuffle_sharer_cache currently expects per-sample batch size 1")
        if receiver_mask is None:
            receiver_mask = torch.ones_like(receiver_ids)
        if teacher_mask is None:
            teacher_mask = torch.ones_like(teacher_ids)

        receiver_length = int(receiver_ids.shape[1])
        original_teacher_length = int(teacher_ids.shape[1])
        if original_teacher_length < receiver_length:
            pad_length = receiver_length - original_teacher_length
            teacher_ids = torch.nn.functional.pad(
                teacher_ids,
                (pad_length, 0),
                value=int(teacher_pad_token_id),
            )
            teacher_mask = torch.nn.functional.pad(
                teacher_mask,
                (pad_length, 0),
                value=0,
            )
        elif original_teacher_length > receiver_length:
            # Match LongBench's existing long-input policy: retain both ends.
            prefix_length = receiver_length // 2
            suffix_length = receiver_length - prefix_length
            teacher_ids = torch.cat(
                [teacher_ids[:, :prefix_length], teacher_ids[:, -suffix_length:]],
                dim=1,
            )
            teacher_mask = torch.cat(
                [teacher_mask[:, :prefix_length], teacher_mask[:, -suffix_length:]],
                dim=1,
            )

        if teacher_ids.shape[1] != receiver_ids.shape[1]:
            raise AssertionError("Shuffled sharer stream was not aligned to receiver length")
        return (
            {
                "input_ids": [receiver_ids, teacher_ids],
                "attention_mask": [receiver_mask, teacher_mask],
            },
            {
                "receiver_cache_length": receiver_length,
                "sharer_original_length": original_teacher_length,
                "sharer_used_length": int(teacher_ids.shape[1]),
            },
        )
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.model_config = config["model"]
        self.output_config = config["output"]
        self.eval_config = config["eval"]
        self.model_name_normalized = str(self.model_config["model_name"]).lower()
        self.is_cachejpeg_model = self.model_name_normalized == "cachejpeg"
        self.is_cachejpeg_rosetta_model = self.model_name_normalized == "cachejpeg_rosetta"
        cachejpeg_rosetta_cfg = self.model_config.get("cachejpeg_rosetta_config") or {}
        ablation_cfg = cachejpeg_rosetta_cfg.get("ablation") or {}
        self.shuffle_sharer_cache = bool(ablation_cfg.get("shuffle_sharer_cache", False))
        self.shuffle_sharer_cache_seed = int(ablation_cfg.get("shuffle_seed", 0))
        if self.shuffle_sharer_cache and not self.is_cachejpeg_rosetta_model:
            raise ValueError(
                "shuffle_sharer_cache is supported only when model_name=cachejpeg_rosetta"
            )
        self.dataset_name = self.eval_config.get("dataset", "mmlu-redux")
        
        # Extract generation config if provided
        self.generation_config = self.model_config.get("generation_config", {})
        
        # Get dataset-specific configuration
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        self.dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Load LongBench prompt formats if needed
        if self.dataset_name == "longbench":
            prompt_format_path = self.eval_config.get("longbench_prompt_format_path", 
                                                    "longbench/config/dataset2prompt.json")
            
            self.dataset_prompt_formats = json.load(open(prompt_format_path, "r", encoding="utf-8-sig"))
            maxlen_format_path = self.eval_config.get("longbench_maxlen_format_path", 
                                                "longbench/config/dataset2maxlen.json")
            self.dataset_maxlen = json.load(open(maxlen_format_path, "r", encoding="utf-8-sig"))
            
            # Remember whether we are evaluating LongBench-E.
            self.is_longbench_e = self.eval_config.get("longbench_e", False)
            subset_cfg = self.eval_config.get("longbench_e_test_subset", {})
            if subset_cfg is None:
                subset_cfg = {}
            if isinstance(subset_cfg, int):
                subset_cfg = {"enabled": True, "size": subset_cfg}
            if not isinstance(subset_cfg, dict):
                raise ValueError(
                    "eval.longbench_e_test_subset must be a mapping, integer, or null."
                )
            self.longbench_e_test_subset_enabled = bool(
                subset_cfg.get("enabled", False)
            )
            self.longbench_e_test_subset_size = int(subset_cfg.get("size", 200))
            self.longbench_e_test_subset_seed = int(subset_cfg.get("seed", 42))
            self.longbench_e_test_subset_ids: Dict[str, set[str]] = {}
            self.longbench_e_test_subset_counts: Dict[str, int] = {}
            if self.longbench_e_test_subset_enabled:
                if not self.is_longbench_e:
                    raise ValueError(
                        "longbench_e_test_subset requires eval.longbench_e=true."
                    )
                if self.longbench_e_test_subset_size <= 0:
                    raise ValueError("longbench_e_test_subset.size must be positive.")
                if int(self.eval_config.get("sample_interval", 1)) != 1:
                    raise ValueError(
                        "longbench_e_test_subset requires sample_interval=1."
                    )
                if self.eval_config.get("limit") is not None:
                    raise ValueError(
                        "Do not combine longbench_e_test_subset with eval.limit."
                    )
            
            # Tokenizer is assigned later inside evaluate_subject.
            self.tokenizer = None
            
        # Setup output directory
        self.output_dir = Path(self.output_config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Debug options
        self.debug_dump_bad_samples = bool(self.eval_config.get("debug_dump_bad_samples", True))
        self.cuda_launch_blocking = bool(self.eval_config.get("cuda_launch_blocking", False))
        
        # Check if using two-stage based on model_name
        self.use_two_stage = self.model_name_normalized in ["two_stage", "two_stage_rosetta"]
        if self.use_two_stage:
            self.context_model_path = self.model_config.get("context_model_path")
            self.background_prompt = self.model_config.get(
                "background_prompt", 
                "Briefly describe the most useful background to solve the problem:\n\n{question}"
            )
            
            if self.model_name_normalized == "two_stage":
                self.answer_model_path = self.model_config.get("answer_model_path")
                print(f"Two-stage mode enabled:")
                print(f"  Context model: {self.context_model_path}")
                print(f"  Answer model: {self.answer_model_path}")
            elif self.model_name_normalized == "two_stage_rosetta":
                self.rosetta_checkpoint_dir = self.model_config.get("rosetta_checkpoint_dir")
                self.rosetta_subfolder = self.model_config.get("rosetta_subfolder", "final")
                self.rosetta_checkpoint_subfolder = self.eval_config.get("rosetta_checkpoint_subfolder", None)
                print(f"Two-stage Rosetta mode enabled:")
                print(f"  Context model: {self.context_model_path}")
                print(f"  Rosetta checkpoint: {self.rosetta_checkpoint_dir}")
                print(f"  Rosetta subfolder: {self.rosetta_subfolder}")
                if self.rosetta_checkpoint_subfolder is not None:
                    print(f"  Rosetta checkpoint subfolder override: {self.rosetta_checkpoint_subfolder}")

        # Resolve local model paths when available; fall back to HF ids otherwise.
        rosetta_cfg = self.model_config.get("rosetta_config", {})
        if isinstance(rosetta_cfg, dict) and rosetta_cfg:
            self.model_config["rosetta_config"]["base_model"] = self._resolve_model_ref(
                rosetta_cfg.get("base_model"),
                ("base_model_path", "base_model_local_dir"),
                rosetta_cfg,
            )
            self.model_config["rosetta_config"]["teacher_model"] = self._resolve_model_ref(
                rosetta_cfg.get("teacher_model"),
                ("teacher_model_path", "teacher_model_local_dir"),
                rosetta_cfg,
            )

        print(f"Evaluating on dataset: {self.dataset_name}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        print(f"Requested GPU IDs: {self.eval_config['gpu_ids']}")
        print(f"Answer method: {self.eval_config['answer_method']}")

    def _make_subject_splits(self, num_gpus: int) -> List[str]:
        """Create virtual subject splits for datasets without native subjects.

        For datasets like math-500 and gsm8k, return SPLIT_i_OF_N identifiers
        so we can distribute the workload across GPUs evenly.
        """
        return [f"SPLIT_{i}_OF_{num_gpus}" for i in range(num_gpus)]

    def _dump_bad_sample(self, subject: str, question_id: int, example: Dict[str, Any], error: Exception, prompt: Optional[str] = None):
        """Dump problematic sample for post-mortem analysis."""
        try:
            dump_dir = self.output_dir / "bad_samples"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"bad_{self.dataset_name}_{subject}_{question_id}.json"
            record = {
                "dataset": self.dataset_name,
                "subject": subject,
                "question_id": question_id,
                "error": str(error),
                "example": example,
            }
            if prompt is not None:
                record["prompt"] = prompt
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            print(f"Saved bad sample to {dump_path}")
        except Exception as e:
            print(f"Failed to dump bad sample for {subject} #{question_id}: {e}")

    def _load_longbench_dataset(self, subject: str):
        """Load a LongBench subject from a local json/jsonl directory or the HF hub."""
        local_data_dir = self.eval_config.get("longbench_local_data_dir", None)
        if local_data_dir:
            normalized_subject = re.sub(r"_e$", "", subject)
            candidate_files = [
                Path(local_data_dir) / f"{subject}.jsonl",
                Path(local_data_dir) / f"{subject}.json",
                Path(local_data_dir) / f"{normalized_subject}.jsonl",
                Path(local_data_dir) / f"{normalized_subject}.json",
            ]
            for file_path in candidate_files:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix.lower() == ".jsonl":
                            rows = [json.loads(line) for line in f if line.strip()]
                        else:
                            rows = json.load(f)

                    if isinstance(rows, dict):
                        for key in ("data", "examples", "items"):
                            if key in rows and isinstance(rows[key], list):
                                rows = rows[key]
                                break
                    if not isinstance(rows, list):
                        raise ValueError(f"Unsupported LongBench local file format: {file_path}")

                    from datasets import Dataset, DatasetDict
                    dataset = Dataset.from_list(rows)
                    return DatasetDict({self.dataset_config["test_split"]: dataset})

        # Fall back to Hugging Face datasets
        return load_dataset(self.dataset_config["dataset_name"], subject)

    def _prepare_longbench_e_test_subset(self, subjects: List[str]) -> None:
        """Select an exact, deterministic held-out subset across LongBench-E subjects."""

        if not getattr(self, "longbench_e_test_subset_enabled", False):
            return
        candidates: list[tuple[int, str, str]] = []
        split_name = self.dataset_config["test_split"]
        for subject in subjects:
            if not str(subject).endswith("_e"):
                raise ValueError(
                    "The LongBench-E test subset can contain only *_e subjects."
                )
            test_data = self._load_longbench_dataset(subject)[split_name]
            for example in test_data:
                example_id = str(example["_id"])
                heldout_hash = int(
                    hashlib.sha256(example_id.encode("utf-8")).hexdigest(), 16
                )
                # Training uses hash % 4 != 1, so the selectable test subset
                # must be drawn exclusively from the complementary partition.
                if heldout_hash % 4 != 1:
                    continue
                rank_key = (
                    f"{self.longbench_e_test_subset_seed}:{subject}:{example_id}"
                )
                rank_hash = int(
                    hashlib.sha256(rank_key.encode("utf-8")).hexdigest(), 16
                )
                candidates.append((rank_hash, str(subject), example_id))

        requested_size = self.longbench_e_test_subset_size
        if len(candidates) < requested_size:
            raise ValueError(
                "LongBench-E held-out partition contains only "
                f"{len(candidates)} samples for the selected subjects, fewer than "
                f"the requested subset size {requested_size}."
            )
        selected = sorted(candidates, key=lambda item: item[0])[:requested_size]
        selected_ids: Dict[str, set[str]] = defaultdict(set)
        for _, subject, example_id in selected:
            selected_ids[subject].add(example_id)
        self.longbench_e_test_subset_ids = dict(selected_ids)
        self.longbench_e_test_subset_counts = {
            subject: len(ids) for subject, ids in selected_ids.items()
        }
        if sum(self.longbench_e_test_subset_counts.values()) != requested_size:
            raise RuntimeError("LongBench-E subset construction did not produce the requested size.")
        print(
            "Prepared deterministic LongBench-E held-out subset: "
            f"size={requested_size}, seed={self.longbench_e_test_subset_seed}, "
            f"per_subject={dict(sorted(self.longbench_e_test_subset_counts.items()))}"
        )

    def _longbench_prediction_split_dir(self, is_longbench_e: bool) -> str:
        if not is_longbench_e:
            return "pred"
        if getattr(self, "longbench_e_test_subset_enabled", False):
            return (
                f"pred_e_subset_{self.longbench_e_test_subset_size}"
                f"_seed_{self.longbench_e_test_subset_seed}"
            )
        return "pred_e"

    @staticmethod
    def _normalize_text(text: Any) -> str:
        if text is None:
            return ""
        text = str(text).lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @classmethod
    def _tokenize_for_score(cls, text: Any) -> List[str]:
        normalized = cls._normalize_text(text)
        return normalized.split() if normalized else []

    @classmethod
    def _f1_score(cls, prediction: Any, ground_truth: Any) -> float:
        pred_tokens = cls._tokenize_for_score(prediction)
        gt_tokens = cls._tokenize_for_score(ground_truth)
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def _exact_match_score(cls, prediction: Any, ground_truth: Any) -> float:
        return float(cls._normalize_text(prediction) == cls._normalize_text(ground_truth))

    @classmethod
    def _lcs_length(cls, a: List[str], b: List[str]) -> int:
        if not a or not b:
            return 0
        dp = [0] * (len(b) + 1)
        for x in a:
            prev = 0
            for j, y in enumerate(b, start=1):
                temp = dp[j]
                if x == y:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[-1]

    @classmethod
    def _rouge_l_score(cls, prediction: Any, ground_truth: Any) -> float:
        pred_tokens = cls._tokenize_for_score(prediction)
        gt_tokens = cls._tokenize_for_score(ground_truth)
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        lcs = cls._lcs_length(pred_tokens, gt_tokens)
        if lcs == 0:
            return 0.0
        precision = lcs / len(pred_tokens)
        recall = lcs / len(gt_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _longbench_metric_type(self, subject: str) -> str:
        subject = re.sub(r"_e$", "", subject)
        rouge_subjects = {"gov_report", "qmsum", "multi_news", "vcsum", "samsum"}
        qa_f1_subjects = {
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "triviaqa"
        }
        exact_subjects = {"trec", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p", "lsht"}
        if subject in rouge_subjects:
            return "rouge_l"
        if subject in qa_f1_subjects:
            return "f1"
        if subject in exact_subjects:
            return "em"
        return "em"

    def _extract_longbench_gold(self, example: Dict[str, Any]) -> str:
        answers = example.get("answers", "")
        if isinstance(answers, list):
            return answers[0] if answers else ""
        return str(answers)

    @staticmethod
    def _longbench_length_bucket(input_length: Optional[int]) -> str:
        """Bucket LongBench samples by actual model input token length."""
        if input_length is None:
            return "unknown"
        try:
            length = int(input_length)
        except (TypeError, ValueError):
            return "unknown"
        if length < 4096:
            return "0-4k"
        if length < 8192:
            return "4k-8k"
        return "8k+"

    def _score_longbench_subject(self, subject: str, output_file: Path) -> Dict[str, Any]:
        metric_type = self._longbench_metric_type(subject)
        scores = []
        rows = []
        bucket_scores: Dict[str, List[float]] = defaultdict(list)
        if not output_file.exists():
            return {
                "subject": subject,
                "metric": metric_type,
                "score": 0.0,
                "num_samples": 0,
                "length_buckets": {},
            }
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                pred = row.get("pred", "")
                gold = row.get("answers", "")
                if isinstance(gold, list):
                    gold_list = gold
                elif isinstance(gold, str):
                    gold_list = [gold]
                else:
                    gold_list = [str(gold)]
                if metric_type == "rouge_l":
                    sample_score = max(self._rouge_l_score(pred, g) for g in gold_list) if gold_list else 0.0
                elif metric_type == "f1":
                    sample_score = max(self._f1_score(pred, g) for g in gold_list) if gold_list else 0.0
                else:
                    sample_score = max(self._exact_match_score(pred, g) for g in gold_list) if gold_list else 0.0
                scores.append(sample_score)
                length_bucket = row.get("length_bucket")
                if length_bucket is None:
                    length_bucket = self._longbench_length_bucket(row.get("input_length", row.get("length")))
                bucket_scores[length_bucket].append(sample_score)
                rows.append({
                    "_id": row.get("_id"),
                    "score": sample_score,
                    "input_length": row.get("input_length"),
                    "length_bucket": length_bucket,
                })
        return {
            "subject": subject,
            "metric": metric_type,
            "score": float(np.mean(scores)) if scores else 0.0,
            "num_samples": len(scores),
            "length_buckets": {
                bucket: {
                    "score": float(np.mean(values)) if values else 0.0,
                    "num_samples": len(values),
                }
                for bucket, values in bucket_scores.items()
            },
            "details": rows,
        }

    def _has_valid_longbench_output(self, output_file: Path) -> bool:
        """Return True only when the subject output file exists and contains at least one non-empty line."""
        if not output_file.exists() or output_file.stat().st_size == 0:
            return False
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        return True
        except Exception:
            return False
        return False
    
    def format_example(self, example: Dict[str, Any], use_cot: bool = True) -> str:
        """
        Format an example into a prompt.
        
        Args:
            example: Example dictionary
            use_cot: Whether to use chain-of-thought prompting
            
        Returns:
            Formatted prompt string
        """
        if self.dataset_name == "mmmlu":
            return self._format_mmmlu_example(example, use_cot)
        elif self.dataset_name == "mmlu-redux":
            return self._format_mmlu_redux_example(example, use_cot)
        elif self.dataset_name == "gpqa":
            return self._format_gpqa_example(example, use_cot)
        elif self.dataset_name in ["math-500", "gsm8k"]:
            return self._format_math_problem_example(example, use_cot)
        elif self.dataset_name == "openbookqa":
            return self._format_openbookqa_example(example, use_cot)
        elif self.dataset_name == "ai2-arc":
            return self._format_ai2_arc_example(example, use_cot)
        elif self.dataset_name == "mmlu-pro":
            return self._format_mmlu_pro_example(example, use_cot)
        elif self.dataset_name == "ceval":
            return self._format_ceval_example(example, use_cot)
        elif self.dataset_name == "longbench":
            return self._format_longbench_example(example)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _format_mmmlu_example(self, example: Dict[str, Any], use_cot: bool, subject: Optional[str] = None, use_template: bool = True) -> str:
        """Format MMMLU example."""
        question_text = example['Question']
        choices = ""
        for i, choice_key in enumerate(['A', 'B', 'C', 'D']):
            if choice_key in example:
                choices += f"{choice_key}. {example[choice_key]}\n"

        # Localized prompt by subject (e.g., SW_KE uses Swahili). Fallback to English otherwise.
        prompt = build_prompt(
            dataset="mmmlu",
            locale=subject or "",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_mmlu_redux_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format MMLU-Redux example using unified prompt builder."""
        # Build choices string (A-D)
        choices = ""
        for i, choice in enumerate(example['choices']):
            choices += f"{chr(65+i)}. {choice}\n"

        # Use shared prompt builder for consistency with MMMLU
        prompt = build_prompt(
            dataset="mmlu-redux",
            locale="",
            question=example['question'],
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    


    def _format_longbench_example(self, example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:


        current_subject = self.current_evaluating_subject  

        subject = re.sub(r"_e$", "", current_subject) if self.is_longbench_e else current_subject
        prompt_format = self.dataset_prompt_formats[subject]
        
        raw_prompt = prompt_format.format(**example)
        
        max_length = self.model_config.get("max_length", 32768)
        tokenized_raw = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_raw) > max_length:
            half_len = int(max_length / 2)
            raw_prompt = tokenizer.decode(tokenized_raw[:half_len], skip_special_tokens=True) + \
                        tokenizer.decode(tokenized_raw[-half_len:], skip_special_tokens=True)
        
        no_chat_template_tasks = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
        if subject not in no_chat_template_tasks:
            messages = [{"role": "user", "content": raw_prompt}]
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            final_prompt = raw_prompt
        
        return final_prompt

    def _prepare_gpqa_item(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare GPQA example into unified fields with deterministic shuffling.

        GPQA columns:
          - Question
          - Correct Answer
          - Incorrect Answer 1/2/3
        Optional revised columns (use if present and not None/empty):
          - extra_revised_question
          - extra_revised_correct_answer
          - extra_revised_incorrect_answer_1/2/3
        """
        # Prefer revised fields if available and non-empty
        def pick(primary_key: str, revised_key: str) -> str:
            revised_val = example.get(revised_key)
            if revised_val is not None and str(revised_val).strip() != "":
                return str(revised_val)
            return str(example.get(primary_key, ""))

        question_text = pick("Question", "Extra Revised Question")
        correct = pick("Correct Answer", "Extra Revised Correct Answer")
        inc1 = pick("Incorrect Answer 1", "Extra Revised Incorrect Answer 1")
        inc2 = pick("Incorrect Answer 2", "Extra Revised Incorrect Answer 2")
        inc3 = pick("Incorrect Answer 3", "Extra Revised Incorrect Answer 3")

        all_choices = [correct, inc1, inc2, inc3]
        # Deterministic shuffle based on content to keep mapping stable across calls
        seed_source = "||".join([question_text] + all_choices)
        seed_int = int(hashlib.md5(seed_source.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed_int)
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [all_choices[idx] for idx in indices]
        correct_new_index = shuffled.index(correct)

        return {
            "question": question_text,
            "choices": shuffled,
            "answer": correct_new_index,
        }

    def _format_gpqa_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format GPQA example using the same prompt template as MMLU-Redux."""
        prepared = self._prepare_gpqa_item(example)
        choices = ""
        for i, choice in enumerate(prepared['choices']):
            choices += f"{chr(65+i)}. {choice}\n"

        prompt = build_prompt(
            dataset="gpqa",  # reuse same template
            locale="",
            question=prepared['question'],
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt

    def _format_math_problem_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format math problem examples (MATH-500, GSM8K) with a shared prompt template."""
        if self.dataset_name == "math-500":
            question_text = example.get('problem', '')
        elif self.dataset_name == "gsm8k":
            question_text = example.get('question', '')
        else:
            question_text = ""
        
        template = (
                "Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n"
                "{question}\n\n"
                "Please think step by step and explain your reasoning. Remember to put your answer on its own line after \"Answer:\", and you do not need to use a \\boxed command."
            )
        return template.replace("{question}", question_text)
    
    def _format_openbookqa_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format OpenBookQA example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question_stem', '')
        # OpenBookQA 'choices' can be either
        # 1) a dict: {'text': [...], 'label': [...]} (HF common form), or
        # 2) a list of dicts: [{'text': str, 'label': 'A'|'B'|...}, ...]
        choices_texts: List[str] = []
        raw_choices = example.get('choices')
        if isinstance(raw_choices, dict):
            choices_texts = list(raw_choices.get('text', []))
        elif isinstance(raw_choices, list):
            for item in raw_choices:
                if isinstance(item, dict):
                    choices_texts.append(str(item.get('text', '')))
                else:
                    choices_texts.append(str(item))
        choices = ""
        for i, text in enumerate(choices_texts):
            choices += f"{chr(65+i)}. {text}\n"

        # Use shared prompt builder for consistency with MMLU
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_ai2_arc_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format AI2-ARC example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question', '')
        # AI2-ARC 'choices' can be either
        # 1) a dict: {'text': [...], 'label': [...]} (HF common form), or
        # 2) a list of dicts: [{'text': str, 'label': 'A'|'B'|...}, ...]
        choices_texts: List[str] = []
        raw_choices = example.get('choices')
        if isinstance(raw_choices, dict):
            choices_texts = list(raw_choices.get('text', []))
        elif isinstance(raw_choices, list):
            for item in raw_choices:
                if isinstance(item, dict):
                    choices_texts.append(str(item.get('text', '')))
                else:
                    choices_texts.append(str(item))
        choices = ""
        for i, text in enumerate(choices_texts):
            choices += f"{chr(65+i)}. {text}\n"

        # Use shared prompt builder for consistency with MMLU
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_mmlu_pro_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format MMLU-Pro example with up to 10 options (A-J)."""
        question_text = example.get('question', '')
        options = example.get('options', [])
        
        # Build choices string (A-J for up to 10 options)
        choices = ""
        for i, option in enumerate(options):
            if i < 10:  # Support up to 10 options (A-J)
                choices += f"{chr(65+i)}. {option}\n"
        
        # Use shared prompt builder for consistency
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def _format_ceval_example(self, example: Dict[str, Any], use_cot: bool, use_template: bool = True) -> str:
        """Format C-EVAL example using the same prompt template as MMLU-Redux."""
        question_text = example.get('question', '')
        
        # Build choices string from A, B, C, D fields
        choices = ""
        for letter in ['A', 'B', 'C', 'D']:
            choice_text = example.get(letter, '')
            if choice_text:
                choices += f"{letter}. {choice_text}\n"
        
        # Use shared prompt builder for consistency
        prompt = build_prompt(
            dataset="mmlu-redux",  # reuse same template
            locale="",
            question=question_text,
            choices=choices,
            use_cot=use_cot,
            use_template=use_template
        )
        return prompt
    
    def parse_answer(self, example: Dict[str, Any]) -> Optional[str]:
        """
        Parse the correct answer from an example.
        
        Args:
            example: Example dictionary
            
        Returns:
            Correct answer letter or None
        """
        if self.dataset_name == "mmmlu":
            answer_key = example.get('Answer')
            if answer_key is None:
                return None
            
            # Convert various answer formats to letter
            if isinstance(answer_key, int):
                return chr(65 + answer_key)  # 0->A, 1->B, 2->C, 3->D
            elif isinstance(answer_key, str) and answer_key in ['0', '1', '2', '3']:
                return chr(65 + int(answer_key))
            elif isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None

        elif self.dataset_name == "mmlu-redux":  # mmlu-redux
            error_type = example.get('error_type', '')
            if error_type in ['no_correct_answer', 'expert']:
                return None
            
            if error_type == 'wrong_groundtruth':
                if example.get('correct_answer') is not None:
                    answer = example['correct_answer']
                    if answer >= '0' and answer <= '3':
                        answer_num = int(answer)
                    else:
                        answer_num = ord(answer) - ord('A')
                else:
                    answer_num = int(example['answer'])
            else:
                answer_num = int(example['answer'])
            
            return chr(65 + answer_num) if answer_num is not None else None
        elif self.dataset_name == "longbench":
            # For LongBench, we don't parse answers as we're generating text
            return None
        elif self.dataset_name == "gpqa":
            # Build deterministic shuffled mapping and return the correct letter
            prepared = self._prepare_gpqa_item(example)
            return chr(65 + int(prepared['answer']))
        elif self.dataset_name == "math-500":
            gt = example.get('answer')
            return None if gt is None else str(gt).strip()
        elif self.dataset_name == "gsm8k":
            full = example.get('answer', '')
            if not isinstance(full, str):
                full = str(full)
            if '####' in full:
                tail = full.split('####')[-1].strip()
                m = re.search(r"[-+]?\d+(?:\.\d+)?", tail)
                return m.group(0) if m else tail
            return None
        elif self.dataset_name == "openbookqa":
            answer_key = example.get('answerKey')
            if answer_key is None:
                return None
            # answerKey should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "ai2-arc":
            answer_key = example.get('answerKey')
            if answer_key is None:
                return None
            # answerKey should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "mmlu-pro":
            answer_key = example.get('answer')
            if answer_key is None:
                return None
            # answer should be A, B, C, D, E, F, G, H, I, or J
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                return answer_key
            else:
                return None
        elif self.dataset_name == "ceval":
            answer_key = example.get('answer')
            if answer_key is None:
                return None
            # answer should be A, B, C, or D
            if isinstance(answer_key, str) and answer_key in ['A', 'B', 'C', 'D']:
                return answer_key
            else:
                return None
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def extract_predicted_answer(self, content: str) -> Optional[str]:
        """Extract model's predicted answer according to the active dataset.

        - math-500: use regex to capture the line after 'Answer:'
        - others: fallback to shared extract_answer_from_content
        """
        if self.dataset_name == "math-500":
            match = re.search(r"(?i)Answer\s*:\s*([^\n]+)", content)
            return match.group(1).strip() if match else None
        return extract_answer_from_content(content)

    def _measure_latency_ms(self, run_fn, device: torch.device) -> Tuple[Any, float]:
        """Measure end-to-end wall latency, including CPU transport and waits.

        Args:
            run_fn: Callable that performs the inference and returns outputs
            device: Torch device used for inference

        Returns:
            (result, latency_ms)
        """
        use_cuda = isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        result = run_fn()
        if use_cuda:
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, float(elapsed_ms)

    def create_segmented_kv_cache_index(self, instruction_length: int, response_length: int, 
                                       proportion: float, order_mode: str, device: torch.device) -> List[torch.Tensor]:
        """
        Create segmented KV cache index for both instruction and response parts, 
        with automatic segmentation based on index value changes.
        
        Args:
            instruction_length: Total number of instruction tokens
            response_length: Total number of response tokens  
            proportion: Proportion of instruction tokens that should use [1, 0] (rest use [-1, 0])
            order_mode: "front" ([1,0] first) or "back" ([1,0] last)
            device: Device to create tensors on
            
        Returns:
            List of tensors, each representing a segment with consistent kv_cache values
        """
        if proportion < 0.0 or proportion > 1.0:
            raise ValueError(f"proportion must be between 0.0 and 1.0, got {proportion}")
        
        if order_mode not in ["front", "back"]:
            raise ValueError(f"order_mode must be 'front' or 'back', got '{order_mode}'")
        
        # Calculate split sizes for instruction
        instruction_positive_length = int(instruction_length * proportion)
        instruction_negative_length = instruction_length - instruction_positive_length
        
        # Create the complete sequence (instruction + response)
        complete_sequence = []
        
        # Add instruction part according to order_mode
        if order_mode == "front":
            # [1, 0] tokens first, then [-1, 0] tokens in instruction
            complete_sequence.extend([[1, 0]] * instruction_positive_length)
            complete_sequence.extend([[-1, 0]] * instruction_negative_length)
        else:  # order_mode == "back"
            # [-1, 0] tokens first, then [1, 0] tokens in instruction
            complete_sequence.extend([[-1, 0]] * instruction_negative_length)
            complete_sequence.extend([[1, 0]] * instruction_positive_length)
        
        # Add response part (always [-1, 0])
        complete_sequence.extend([[-1, 0]] * response_length)
        
        # Now segment the complete sequence based on value changes
        if len(complete_sequence) == 0:
            return []
        
        segments = []
        current_segment_start = 0
        current_value = complete_sequence[0]
        
        for i in range(1, len(complete_sequence)):
            if complete_sequence[i] != current_value:
                # Found a change, create segment for previous section
                segment_length = i - current_segment_start
                segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
                segments.append(segment)
                
                # Update for next segment
                current_segment_start = i
                current_value = complete_sequence[i]
        
        # Handle the last segment
        segment_length = len(complete_sequence) - current_segment_start
        segment = torch.tensor(current_value, dtype=torch.long).repeat(segment_length, 1).unsqueeze(0).to(device)
        segments.append(segment)
        
        return segments

    def prepare_model_inputs(self, prompt: str, tokenizer, device: torch.device,
                              model_type: str, llm_tokenizer: Optional[Any],
                              answer_method: str, proportion: float = 1.0, 
                              order_mode: str = "front"):
        """
        Prepare model inputs (input_ids, attention_mask, position_ids, kv_cache_index) for
        both HF and Rosetta models, separated from the generation stage.

        Args:
            proportion: Float between 0.0 and 1.0 controlling the proportion of instruction 
                       tokens that should use [1, 0] vs [-1, 0] (default: 1.0, all [1, 0])
            order_mode: String specifying order of mixed instruction indices:
                       "front" - [1, 0] tokens first, then [-1, 0] tokens
                       "back" - [-1, 0] tokens first, then [1, 0] tokens

        Returns a dict with keys:
        - input_ids
        - attention_mask
        - position_ids
        - kv_cache_index
        - printable_text (str): chat-formatted input text for logging
        """
        messages = [{"role": "user", "content": prompt}]

        use_aligner = (
            model_type in {"rosetta", "cachejpeg_rosetta"}
            and llm_tokenizer is not None
        )

        # Build chat-formatted text
        if not use_aligner:
            if answer_method == 'logits':
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                # Use custom response text if provided, otherwise default
                response_text = self.eval_config.get("response_text", "The correct answer is")
                text += response_text
                response_length = tokenizer(response_text, add_special_tokens=False).input_ids.__len__()
            else: # generate
                
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                response_length = 1
            # Default HF/Qwen path (and Rosetta generate path)
            tokenized = tokenizer(text, return_tensors="pt").to(device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            outputs = {
                "inputs": {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                },
                "printable_text": text
            }

            if model_type == "rosetta":
                full_length = input_ids.shape[1]
                instruction_length = full_length - response_length
                
                # Create segmented KV cache index for the complete sequence
                kv_cache_list = self.create_segmented_kv_cache_index(
                    instruction_length=instruction_length,
                    response_length=response_length,
                    proportion=proportion,
                    order_mode=order_mode,
                    device=device
                )

                if attention_mask is None:
                    outputs['inputs']["position_ids"] = torch.arange(input_ids.shape[-1], dtype=torch.long).unsqueeze(0).to("cuda")
                else:
                    outputs['inputs']["position_ids"] = attention_mask.long().cumsum(-1) - 1
                outputs['inputs']['kv_cache_index'] = kv_cache_list
            
        # Rosetta logits path with alignment (dual tokenizers)
        # TODO: add rosetta proportion for aligner
        if use_aligner:
            alignment_strategy = self.model_config["rosetta_config"].get("alignment_strategy", "prefix")
            aligner = TokenAligner(
                slm_tokenizer=tokenizer,
                llm_tokenizer=llm_tokenizer,
                strategy=AlignmentStrategy(alignment_strategy)
            )

            if answer_method == 'logits':
                # Use custom response text if provided, otherwise default
                response_text = self.eval_config.get("response_text", "The correct answer is")
                messages.append({"role": "assistant", "content": response_text})
                remove_last_surfix = True
                add_generation_prompt = False
            else: # generate
                remove_last_surfix = False
                add_generation_prompt = True

            details = aligner.align_chat_messages(
                messages,
                add_generation_prompt=add_generation_prompt,
                return_details=True,
                enable_thinking=False,
                remove_last_surfix=remove_last_surfix
            )

            slm_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0).to(device)
            llm_ids = torch.tensor(details['llm_ids_padded']).unsqueeze(0).to(device)

            assert slm_ids.shape == llm_ids.shape, f"SLM and LLM input lengths do not match: {slm_ids.shape} vs {llm_ids.shape}"

            slm_pad_mask = torch.tensor(details['slm_padding_mask']).unsqueeze(0)
            llm_pad_mask = torch.tensor(details['llm_padding_mask']).unsqueeze(0)

            slm_attention_mask = (~slm_pad_mask).float()
            llm_attention_mask = (~llm_pad_mask).float()

            message_mask = torch.tensor(details['message_mask'])
            
            # Create kv_cache_index and split by message_mask transitions in one pass
            kv_cache_list = []
            start = 0
            current_value = message_mask[0].item()
            
            for j in range(1, len(message_mask)):
                if message_mask[j] != message_mask[j - 1]:
                    # Found a change point, create segment for previous section
                    segment_length = j - start
                    if current_value:
                        segment = torch.tensor([1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
                    else:
                        segment = torch.tensor([-1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
                    kv_cache_list.append(segment)
                    
                    start = j
                    current_value = message_mask[j].item()
            
            # Handle the last segment
            segment_length = len(message_mask) - start
            if current_value:
                segment = torch.tensor([1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
            else:
                segment = torch.tensor([-1, 0]).repeat(segment_length, 1).unsqueeze(0).to(device)
            kv_cache_list.append(segment)

            input_ids = [slm_ids, llm_ids]
            attention_mask = [slm_attention_mask.to(device), llm_attention_mask.to(device)]
            position_ids = torch.arange(slm_ids.shape[1]).unsqueeze(0).to(device)

            aligned_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            # RosettaModel consumes segmentation metadata. CacheJPEG-Rosetta
            # fuses the complete aligned teacher cache and only needs the two
            # model-specific aligned input streams.
            if model_type == "rosetta":
                aligned_inputs["position_ids"] = position_ids
                aligned_inputs["kv_cache_index"] = kv_cache_list
            outputs = {
                "inputs": aligned_inputs,
                "printable_text": (details["slm_text"], details["llm_text"])
            }

        return outputs

    @torch.no_grad()
    def evaluate_subject(self, subject: str, model, tokenizer, device: torch.device, 
                        model_type: str = "hf", llm_tokenizer: Optional[Any] = None) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray], List, List]:
        """
        Evaluate model on a specific subject.
        
        Args:
            subject: Subject name
            model: Model object
            tokenizer: Tokenizer object
            device: Device to run on
            model_type: Type of model
            
        Returns:
            Tuple of (correctness_array, accuracy, probabilities, length_stats, cot_logs)
        """
        # Load dataset
        is_virtual_split = False
        split_index = 0
        total_splits = 1
        if self.dataset_name in ["math-500", "gsm8k", "openbookqa", "gpqa", "ai2-arc", "mmlu-pro"]:
            # Detect virtual split subject: SPLIT_i_OF_N
            m = re.match(r"^SPLIT_(\d+)_OF_(\d+)$", str(subject))
            if m:
                is_virtual_split = True
                split_index = int(m.group(1))
                total_splits = max(1, int(m.group(2)))

        if self.dataset_name == "math-500":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "gsm8k":
            base_subset = "main" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "openbookqa":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "gpqa":
            base_subset = "gpqa_diamond" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "ai2-arc":
            base_subset = "ARC-Challenge" if is_virtual_split else subject
            dataset = load_dataset(self.dataset_config["dataset_name"], base_subset)
        elif self.dataset_name == "mmlu-pro":
            dataset = load_dataset(self.dataset_config["dataset_name"])
        elif self.dataset_name == "ceval":
            dataset = load_dataset(self.dataset_config["dataset_name"], subject)
        elif self.dataset_name == "longbench":
            dataset = self._load_longbench_dataset(subject)
        else:
            dataset = load_dataset(self.dataset_config["dataset_name"], subject)
        # dataset = load_from_disk("local/teacher_datasets/MMMLU")
        test_data = dataset[self.dataset_config["test_split"]]
        
        self.current_evaluating_subject = subject
        # Store the tokenizer on the evaluator instance for later use.
        self.tokenizer = tokenizer
 
        # For LongBench, we don't use option token IDs
        num_options = 10 if self.dataset_name == "mmlu-pro" else 4
        if self.dataset_name != "longbench":
            option_ids = get_option_token_ids(tokenizer, num_options)
        
        # Prepare rule-based math evaluators if needed
        rule_evaluator = None
        if self.dataset_name == "gsm8k":
            rule_evaluator = GSM8KEvaluator()
        elif self.dataset_name == "math-500":
            rule_evaluator = MATH500Evaluator()

        cors = []
        all_probs = []
        length_stats = []
        cot_logs = []
        total_count = 0
        skip_count = 0
        printed_example = False

        if self.dataset_name == "longbench":
            # Check whether this subject uses the LongBench-E suffix.
            is_longbench_e = subject.endswith("_e")
            
            # Choose the output base directory based on the subject type.
            if is_longbench_e:
                # LongBench-E subjects are saved under pred_e/ and drop the _e suffix.
                subject_name = subject[:-2]
                output_base_dir = self.output_dir / self._longbench_prediction_split_dir(True) / self.model_config["model_name"].split("/")[-1]
            else:
                # Standard LongBench subjects are saved under pred/.
                subject_name = subject
                output_base_dir = self.output_dir / "pred" / self.model_config["model_name"].split("/")[-1]
            
            output_base_dir.mkdir(parents=True, exist_ok=True)
            # Use the normalized subject name as the output filename.
            output_file = output_base_dir / f"{subject_name}.jsonl"
            
            # Remove any previous output so each run starts cleanly.
            if output_file.exists():
                output_file.unlink()

        # Sampling configuration
        sample_interval = self.eval_config.get("sample_interval", 1)
        sample_indices = list(range(0, len(test_data), sample_interval))

        if (
            self.dataset_name == "longbench"
            and is_longbench_e
            and getattr(self, "longbench_e_test_subset_enabled", False)
        ):
            if not self.longbench_e_test_subset_ids:
                raise RuntimeError(
                    "LongBench-E test subset was not prepared before subject evaluation."
                )
            selected_ids = self.longbench_e_test_subset_ids.get(str(subject), set())
            sample_indices = [
                index
                for index in sample_indices
                if str(test_data[index]["_id"]) in selected_ids
            ]

        # Apply virtual split window for datasets without native subjects
        if is_virtual_split and total_splits > 1:
            n = len(test_data)
            start = (split_index * n) // total_splits
            end = ((split_index + 1) * n) // total_splits
            sample_indices = [i for i in sample_indices if start <= i < end]
        limit = self.eval_config.get("limit", None)
        if isinstance(limit, int) and limit > 0:
            # Use first N indices
            sample_indices = sample_indices[:limit]
        elif isinstance(limit, (list, tuple)) and len(limit) == 2:
            # Treat as [start, end) range on original indices
            start, end = limit
            start = 0 if start is None else int(start)
            end = len(test_data) if end is None else int(end)
            sample_indices = [i for i in sample_indices if start <= i < end]

        sharer_shuffle_pairs: Dict[int, int] = {}
        if self.shuffle_sharer_cache:
            if self.dataset_name != "longbench":
                raise ValueError("shuffle_sharer_cache is currently supported only for LongBench")
            if model_type != "cachejpeg_rosetta":
                raise ValueError(
                    "shuffle_sharer_cache requires the CacheJPEG-Rosetta wrapper"
                )
            if self.eval_config["answer_method"] != "generate":
                raise ValueError("shuffle_sharer_cache requires answer_method=generate")
            retained_indices = []
            for sample_idx in sample_indices:
                retained_example = test_data[sample_idx]
                retained_hash = int(
                    hashlib.sha256(str(retained_example["_id"]).encode("utf-8")).hexdigest(),
                    16,
                )
                if retained_hash % 4 == 1:
                    retained_indices.append(sample_idx)
            sharer_shuffle_pairs = self._build_sharer_shuffle_pairs(
                retained_indices,
                seed=self.shuffle_sharer_cache_seed,
                subject=subject,
            )
        
        for i in tqdm(sample_indices, desc=f"Evaluating {subject} ({self.eval_config['answer_method']})"):
            try:
                example = test_data[i]
                sharer_shuffle_metadata = None
                
                if self.dataset_name != "longbench":
                    true_answer = self.parse_answer(example)
                    if true_answer is None:
                        skip_count += 1
                        continue
                else:
                    
                    id_hash=int(hashlib.sha256(str(example["_id"]).encode('utf-8')).hexdigest(), 16)
                    
                    if id_hash%4!=1:
                        skip_count += 1
                        continue

                # Format prompt (pass subject for locale-aware templates)
                if self.dataset_name == "mmmlu":
                    prompt = self._format_mmmlu_example(example, use_cot=self.eval_config["use_cot"], subject=subject, use_template=self.eval_config["use_template"])
                elif self.dataset_name == "mmlu-redux":
                    prompt = self._format_mmlu_redux_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "gpqa":
                    prompt = self._format_gpqa_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name in ["math-500", "gsm8k"]:
                    prompt = self._format_math_problem_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "openbookqa":
                    prompt = self._format_openbookqa_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "ai2-arc":
                    prompt = self._format_ai2_arc_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "mmlu-pro":
                    prompt = self._format_mmlu_pro_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "ceval":
                    prompt = self._format_ceval_example(example, use_cot=self.eval_config["use_cot"], use_template=self.eval_config["use_template"])
                elif self.dataset_name == "longbench":
                    prompt = self._format_longbench_example(example, tokenizer) 
                else:
                    raise ValueError(f"Unknown dataset: {self.dataset_name}")
                
                # Generate answer
                if model_type in ["two_stage", "two_stage_rosetta"]:
                    # Two-stage inference mode (both regular and Rosetta)
                    # Extract question without options
                    if self.dataset_name == "mmmlu":
                        question_text = example.get('Question', '')
                    elif self.dataset_name == "mmlu-redux":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "gpqa":
                        question_text = self._prepare_gpqa_item(example)['question']
                    elif self.dataset_name == "gsm8k":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "math-500":
                        question_text = example.get('problem', '')
                    elif self.dataset_name == "openbookqa":
                        question_text = example.get('question_stem', '')
                    elif self.dataset_name == "ai2-arc":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "mmlu-pro":
                        question_text = example.get('question', '')
                    elif self.dataset_name == "ceval":
                        question_text = example.get('question', '')
                    else:
                        question_text = ""

                    prompt_with_options = prompt

                    if self.eval_config["answer_method"] == 'logits':
                        # Forward logits path
                        response_text = self.eval_config.get("response_text", "The correct answer is")

                        def _two_stage_forward_call():
                            return model.logits_with_context(
                                question_without_options=question_text,
                                question_with_options=prompt_with_options,
                                response_text=response_text
                            )
                        (outputs, bg_context), latency_ms = self._measure_latency_ms(_two_stage_forward_call, device)

                        # Get option token IDs from the tokenizer used for second stage
                        num_options = 10 if self.dataset_name == "mmlu-pro" else 4
                        option_ids = get_option_token_ids(tokenizer, num_options)
                        logits = outputs.logits[0, -1]
                        option_logits = torch.tensor([
                            logits[option_ids[i]].item() for i in range(num_options)
                        ])
                        probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
                        pred = chr(65 + np.argmax(probs))

                        # Record background context as CoT in logits+two-stage
                        cot_text = f"[Background Context]:\n{bg_context}"
                        cot_pred = None
                        input_length, gen_length = None, None
                        cot_input_len, cot_gen_len = None, None
                    else:
                        # Generate using two-stage model
                        def _two_stage_call():
                            return model.generate(
                                question_without_options=question_text,
                                question_with_options=prompt_with_options,
                                communication_max_new_tokens=self.eval_config.get("communication_max_new_tokens", 1024),
                                response_max_new_tokens=self.eval_config.get("response_max_new_tokens", 1024)
                            )
                        content, latency_ms = self._measure_latency_ms(_two_stage_call, device)

                        # Extract and grade answer
                        pred = None
                        math_eval = None
                        if self.dataset_name in ["math-500", "gsm8k"] and rule_evaluator is not None:
                            is_corr, extracted = rule_evaluator.rule_judge(content, true_answer, finish_generation=True)
                            math_eval = {"is_correct": bool(is_corr), "extracted_answer": str(extracted)}
                            pred = str(extracted)
                        else:
                            pred = extract_answer_from_content(content)
                        probs = np.array([0.25, 0.25, 0.25, 0.25])

                        # Get context for logging using process method
                        result = model.process(question_text, prompt_with_options)
                        cot_text = f"[Background Context]:\n{result['context']}\n\n[Answer]:\n{content}"
                        cot_pred = pred
                        # Tokenize to get accurate token counts
                        input_length = len(tokenizer.encode(prompt_with_options, add_special_tokens=False))
                        gen_length = len(tokenizer.encode(content, add_special_tokens=False)) + len(tokenizer.encode(result['context'], add_special_tokens=False))
                        cot_input_len, cot_gen_len = input_length, gen_length
                        
                else:
                    # Regular single-model inference
                    # Prepare the inputs (separated from generation)
                    # Get proportion and order_mode from config, with defaults
                    proportion = self.eval_config.get("kv_cache_proportion", 1.0)
                    order_mode = self.eval_config.get("kv_cache_order_mode", "front")
                    
                    prepared = self.prepare_model_inputs(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        device=device,
                        model_type=model_type,
                        llm_tokenizer=llm_tokenizer,
                        answer_method=self.eval_config["answer_method"],
                        proportion=proportion,
                        order_mode=order_mode
                    )
                    if self.shuffle_sharer_cache:
                        sharer_sample_index = sharer_shuffle_pairs[i]
                        sharer_example = test_data[sharer_sample_index]
                        sharer_prompt = self._format_longbench_example(
                            sharer_example, tokenizer
                        )
                        prepared_sharer = self.prepare_model_inputs(
                            prompt=sharer_prompt,
                            tokenizer=tokenizer,
                            device=device,
                            model_type=model_type,
                            llm_tokenizer=llm_tokenizer,
                            answer_method=self.eval_config["answer_method"],
                            proportion=proportion,
                            order_mode=order_mode,
                        )
                        teacher_tokenizer = (
                            llm_tokenizer if llm_tokenizer is not None else tokenizer
                        )
                        teacher_pad_token_id = teacher_tokenizer.pad_token_id
                        if teacher_pad_token_id is None:
                            teacher_pad_token_id = teacher_tokenizer.eos_token_id
                        if teacher_pad_token_id is None:
                            raise ValueError(
                                "A teacher pad/eos token is required for sharer cache shuffling"
                            )
                        shuffled_inputs, shuffle_lengths = (
                            self._compose_shuffled_sharer_inputs(
                                prepared["inputs"],
                                prepared_sharer["inputs"],
                                teacher_pad_token_id=int(teacher_pad_token_id),
                            )
                        )
                        prepared["inputs"] = shuffled_inputs
                        sharer_shuffle_metadata = {
                            "enabled": True,
                            "receiver_sample_index": int(i),
                            "receiver_sample_id": str(example["_id"]),
                            "sharer_sample_index": int(sharer_sample_index),
                            "sharer_sample_id": str(sharer_example["_id"]),
                            "seed": int(self.shuffle_sharer_cache_seed),
                            **shuffle_lengths,
                        }
                    
                    if self.eval_config["answer_method"] == 'logits':
                        # Forward for logits
                        def _forward_call():
                            return model.forward(**prepared['inputs'])
                        outputs, latency_ms = self._measure_latency_ms(_forward_call, device)

                        logits = outputs.logits[0, -1]
                        option_logits = torch.tensor([
                            logits[option_ids[i]].item() for i in range(num_options)
                        ])
                        probs = torch.nn.functional.softmax(option_logits, dim=0).numpy()
                        pred = chr(65 + np.argmax(probs))

                        # No CoT generation in logits mode
                        input_length, gen_length = None, None
                        cot_pred, cot_input_len, cot_gen_len, cot_text = None, None, None, None
                    elif self.eval_config["answer_method"] == "generate":  # generate
                        # Ensure model has uniform generation config applied
                        #apply_generation_config(model, self.generation_config)

                        inputs = prepared['inputs']
                        def _generate_call():
                            return model.generate(**inputs, **self.generation_config)
                        outputs, latency_ms = self._measure_latency_ms(_generate_call, device)
                        
                        if isinstance(model, RosettaModel):
                            generated_ids = outputs[0]
                            if isinstance(prepared["inputs"]["input_ids"], list):
                                input_length = prepared["inputs"]["input_ids"][0].shape[1]
                            else:
                                input_length = prepared["inputs"]["input_ids"].shape[1]
                            generated_ids = generated_ids[input_length:]

                        else:
                            prepared_input_ids = prepared['inputs']["input_ids"]
                            base_input_ids = (
                                prepared_input_ids[0]
                                if isinstance(prepared_input_ids, list)
                                else prepared_input_ids
                            )
                            input_length = base_input_ids.shape[1]
                            generated_ids = outputs[0][input_length:]
                        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
                        # Default values for non-MATH datasets
                        pred = None
                        math_eval = None
                        if self.dataset_name in ["math-500", "gsm8k"] and rule_evaluator is not None:
                            is_corr, extracted = rule_evaluator.rule_judge(content, true_answer, finish_generation=True)
                            math_eval = {"is_correct": bool(is_corr), "extracted_answer": str(extracted)}
                            pred = str(extracted)
                        else:
                            pred = self.extract_predicted_answer(content)
                        probs = np.array([0.25, 0.25, 0.25, 0.25])
                        gen_length = generated_ids.shape[0]
                        cot_text = content
                        cot_pred = pred
                        cot_input_len, cot_gen_len = input_length, gen_length
                    else:
                        raise ValueError(f"Unknown answer method: {self.eval_config['answer_method']}")
                    
                # Print one example of chat-formatted input (and output if generation) per subject
                if not printed_example:
                    try:
                        if model_type in ["two_stage", "two_stage_rosetta"]:
                            text = prompt  # Just show the formatted prompt for two-stage
                        else:
                            text = prepared.get("printable_text", "")
                        print("\n================ Example IO ({}) ================".format(subject))
                        if isinstance(text, (tuple, list)):
                            try:
                                slm_text, llm_text = text
                                print("[Input with chat template - SLM]:\n" + str(slm_text))
                                print("[Input with chat template - LLM]:\n" + str(llm_text))
                            except Exception:
                                print("[Input with chat template]:\n" + str(text))
                        else:
                            print("[Input with chat template]:\n" + str(text))
                        if self.eval_config["answer_method"] == 'generate' and cot_text is not None:
                            print("\n[Generated output]:\n" + str(cot_text))
                        print("================ End Example IO ================\n")
                    except Exception as e:
                        print(f"Failed to print example IO for {subject}: {e}")
                    finally:
                        printed_example = True
                    
                    # Check correctness
                if self.dataset_name != "longbench":
                    if self.dataset_name in ["math-500", "gsm8k"]:
                        # Use evaluate_answer result if available (generate path). If not, fallback to simple match
                        if 'math_eval' in locals() and math_eval is not None:
                            is_correct = bool(math_eval.get('is_correct', False))
                        else:
                            is_correct = (pred == true_answer) if pred else False
                    else:
                        is_correct = (pred == true_answer) if pred else False
                    cors.append(is_correct)
                    all_probs.append(probs)
                else:
                    is_correct = None
                    
                # Collect length statistics
                if self.eval_config["answer_method"] == 'generate' and input_length is not None and gen_length is not None:
                    length_ratio = gen_length / input_length if input_length > 0 else 0
                    length_stats.append({
                        'subject': subject,
                        'question_id': i,
                        'input_length': input_length,
                        'gen_length': gen_length,
                        'length_ratio': length_ratio,
                        'is_correct': is_correct,
                        'pred': pred,
                        'true_answer': true_answer if self.dataset_name != "longbench" else None
                    })
            # Save LongBench predictions immediately for post-processing.
                if self.dataset_name == "longbench":
                    length_bucket = self._longbench_length_bucket(input_length)
                    output_entry = {
                        "pred": content, 
                        "answers": example["answers"],
                        "all_classes": example["all_classes"],
                        "length": example["length"],
                        "input_length": int(input_length) if input_length is not None else None,
                        "gen_length": int(gen_length) if gen_length is not None else None,
                        "length_bucket": length_bucket,
                        "_id": example["_id"],
                        
                    }
                    if getattr(self, "longbench_e_test_subset_enabled", False):
                        output_entry["longbench_e_test_subset"] = {
                            "size": self.longbench_e_test_subset_size,
                            "seed": self.longbench_e_test_subset_seed,
                        }
                    cachejpeg_stats = getattr(model, "last_codec_stats", None)
                    if cachejpeg_stats is not None:
                        output_entry["cachejpeg_stats"] = dict(cachejpeg_stats)
                    fusion_stats = getattr(model, "last_fusion_stats", None)
                    if fusion_stats is not None:
                        output_entry["fusion_stats"] = dict(fusion_stats)
                    if sharer_shuffle_metadata is not None:
                        output_entry["sharer_cache_shuffle"] = dict(
                            sharer_shuffle_metadata
                        )
                    output_entry["end_to_end_latency_ms"] = (
                        float(latency_ms) if latency_ms is not None else None
                    )
                    transport_stats = getattr(model, "last_transport_stats", None)
                    if transport_stats is not None:
                        output_entry["transport_stats"] = dict(vars(transport_stats))
                
                # Append the prediction to the subject output file.
                    with open(output_file, "a", encoding='utf-8') as f:
                        json.dump(output_entry, f, ensure_ascii=False)
                        f.write('\n')
                        
                # Collect CoT logs
                cot_log_entry = {
                    'subject': subject,
                    'question_id': i,
                    'true_answer': true_answer if self.dataset_name != "longbench" else None,
                    'pred': pred,
                    'is_correct': is_correct,
                    'answer_method': self.eval_config.get('answer_method', ''),
                    'cot_pred': cot_pred,
                    'cot_input_length': cot_input_len,
                    'cot_gen_length': cot_gen_len,
                    'cot_output': cot_text,
                    'answer_latency_ms': float(latency_ms) if 'latency_ms' in locals() and latency_ms is not None else None
                }
                if self.dataset_name == "longbench" and cachejpeg_stats is not None:
                    cot_log_entry["cachejpeg_stats"] = dict(cachejpeg_stats)
                    if transport_stats is not None:
                        cot_log_entry["transport_stats"] = dict(vars(transport_stats))
                
                # Add question and choices based on dataset format
                if self.dataset_name == "mmmlu":
                    cot_log_entry.update({
                        'question': example.get('Question', ''),
                        'A': example.get('A', ''),
                        'B': example.get('B', ''),
                        'C': example.get('C', ''),
                        'D': example.get('D', '')
                    })
                elif self.dataset_name == "gpqa":
                    prepared_gpqa = self._prepare_gpqa_item(example)
                    choices = prepared_gpqa.get('choices', [])
                    cot_log_entry.update({
                        'question': prepared_gpqa.get('question', ''),
                        'A': choices[0] if len(choices) > 0 else '',
                        'B': choices[1] if len(choices) > 1 else '',
                        'C': choices[2] if len(choices) > 2 else '',
                        'D': choices[3] if len(choices) > 3 else ''
                    })
                elif self.dataset_name == "math-500":
                    cot_log_entry.update({
                        'question': example.get('problem', ''),
                        'A': '', 'B': '', 'C': '', 'D': ''
                    })
                    # Add extraction diagnostics from math evaluator if available
                    if 'math_eval' in locals() and math_eval is not None:
                        cot_log_entry.update({
                            'extraction_method_used': math_eval.get('extraction_method_used', ''),
                            'ground_truth_normalized': math_eval.get('ground_truth_normalized', ''),
                            'extracted_normalized': math_eval.get('extracted_normalized', '')
                        })
                elif self.dataset_name == "gsm8k":
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': '', 'B': '', 'C': '', 'D': ''
                    })
                    if 'math_eval' in locals() and math_eval is not None:
                        cot_log_entry.update({
                            'extraction_method_used': math_eval.get('extraction_method_used', ''),
                            'ground_truth_normalized': math_eval.get('ground_truth_normalized', ''),
                            'extracted_normalized': math_eval.get('extracted_normalized', '')
                        })
                elif self.dataset_name == "openbookqa":
                    # Normalize OpenBookQA choices to texts list
                    choices_texts: List[str] = []
                    raw_choices = example.get('choices')
                    if isinstance(raw_choices, dict):
                        choices_texts = list(raw_choices.get('text', []))
                    elif isinstance(raw_choices, list):
                        for item in raw_choices:
                            if isinstance(item, dict):
                                choices_texts.append(str(item.get('text', '')))
                            else:
                                choices_texts.append(str(item))
                    cot_log_entry.update({
                        'question': example.get('question_stem', ''),
                        'A': choices_texts[0] if len(choices_texts) > 0 else '',
                        'B': choices_texts[1] if len(choices_texts) > 1 else '',
                        'C': choices_texts[2] if len(choices_texts) > 2 else '',
                        'D': choices_texts[3] if len(choices_texts) > 3 else ''
                    })
                elif self.dataset_name == "ai2-arc":
                    # Normalize AI2-ARC choices to texts list
                    choices_texts: List[str] = []
                    raw_choices = example.get('choices')
                    if isinstance(raw_choices, dict):
                        choices_texts = list(raw_choices.get('text', []))
                    elif isinstance(raw_choices, list):
                        for item in raw_choices:
                            if isinstance(item, dict):
                                choices_texts.append(str(item.get('text', '')))
                            else:
                                choices_texts.append(str(item))
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': choices_texts[0] if len(choices_texts) > 0 else '',
                        'B': choices_texts[1] if len(choices_texts) > 1 else '',
                        'C': choices_texts[2] if len(choices_texts) > 2 else '',
                        'D': choices_texts[3] if len(choices_texts) > 3 else ''
                    })
                elif self.dataset_name == "mmlu-pro":
                    # MMLU-Pro supports up to 10 options (A-J)
                    options = example.get('options', [])
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': options[0] if len(options) > 0 else '',
                        'B': options[1] if len(options) > 1 else '',
                        'C': options[2] if len(options) > 2 else '',
                        'D': options[3] if len(options) > 3 else '',
                        'E': options[4] if len(options) > 4 else '',
                        'F': options[5] if len(options) > 5 else '',
                        'G': options[6] if len(options) > 6 else '',
                        'H': options[7] if len(options) > 7 else '',
                        'I': options[8] if len(options) > 8 else '',
                        'J': options[9] if len(options) > 9 else ''
                    })
                elif self.dataset_name == "ceval":
                    # C-EVAL uses A, B, C, D fields directly
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': example.get('A', ''),
                        'B': example.get('B', ''),
                        'C': example.get('C', ''),
                        'D': example.get('D', ''),
                    })
                elif self.dataset_name == "mmlu-redux":  # mmlu-redux
                    choices = example.get('choices', [])
                    cot_log_entry.update({
                        'question': example.get('question', ''),
                        'A': choices[0] if len(choices) > 0 else '',
                        'B': choices[1] if len(choices) > 1 else '',
                        'C': choices[2] if len(choices) > 2 else '',
                        'D': choices[3] if len(choices) > 3 else ''
                    })
                elif self.dataset_name == "longbench":
                    cot_log_entry.update({
                        'context': example.get('context', ''),
                        'question': example.get('question', ''),
                        'input': example.get('input', ''),
                        'answers': example.get('answers', []),
                        'all_classes': example.get('all_classes', []),
                        'length': example.get('length', 0),
                        '_id': example.get('_id', f"{subject}_{i}")
                    })
                
                cot_logs.append(cot_log_entry)
                total_count += 1
                
            except Exception as e:
                print(f"Error processing question {i} in subject {subject}: {e}")
                if self.debug_dump_bad_samples:
                    try:
                        # Attempt to include the last built prompt if available
                        maybe_prompt = locals().get('prompt', None)
                        self._dump_bad_sample(subject, i, example, e, maybe_prompt)
                    except Exception as ee:
                        print(f"Failed to record bad sample for {subject} #{i}: {ee}")
                # If CUDA device-side assert, force sync to get accurate site and re-raise
                if "device-side assert" in str(e).lower() and torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                skip_count += 1
                continue
        
        if total_count > 0 and self.dataset_name != "longbench":
            acc = np.mean(cors)
            print(f"{subject} accuracy: {acc*100:.2f}% (evaluated on {total_count} samples, skipped {skip_count})")
        else:
            acc = 0
            print(f"{subject} processed {total_count} samples, skipped {skip_count}")
        
        return np.array(cors) if cors else None, acc, np.array(all_probs) if all_probs else None, length_stats, cot_logs
    
    def evaluate_on_gpu(self, rank: int, gpu_id: int, subjects: List[str], return_dict):
        """
        Evaluate on a single GPU.
        
        Args:
            rank: Process rank
            gpu_id: GPU ID
            subjects: List of subjects to evaluate
            return_dict: Shared dictionary for results
        """
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # Load model
        if "two_stage_rosetta" == self.model_config["model_name"].lower():
            rosetta_subfolder = self.rosetta_checkpoint_subfolder or self.rosetta_subfolder
            model = TwoStageRosetta(
                context_model_path=self.context_model_path,
                rosetta_checkpoint_dir=self.rosetta_checkpoint_dir,
                rosetta_subfolder=rosetta_subfolder,
                device=device,
                max_new_tokens=self.generation_config.get("max_new_tokens", self.eval_config.get("max_new_tokens", 2048)),
                background_prompt=self.background_prompt,
                generation_config=self.generation_config
            )
            # Use the Rosetta tokenizer for consistency
            tokenizer = model.rosetta_tokenizer
            model_type = "two_stage_rosetta"
            llm_tokenizer = model.llm_tokenizer
            print(f"Initialized TwoStageRosetta pipeline on GPU {gpu_id}")
        elif "two_stage" == self.model_config["model_name"].lower():
            model = TwoStageInference(
                context_model_path=self.context_model_path,
                answer_model_path=self.answer_model_path,
                device=device,
                max_new_tokens=self.generation_config.get("max_new_tokens", self.eval_config.get("max_new_tokens", 2048)),
                background_prompt=self.background_prompt,
                generation_config=self.generation_config
            )
            # Use the answer model's tokenizer for consistency
            tokenizer = AutoTokenizer.from_pretrained(self.answer_model_path)
            model_type = "two_stage"
            llm_tokenizer = None
            print(f"Initialized two-stage pipeline on GPU {gpu_id}")
        elif self.is_cachejpeg_rosetta_model:
            model, tokenizer = load_cachejpeg_rosetta_model(
                self.model_config,
                self.eval_config,
                device=device,
                generation_config=self.generation_config,
            )
            model_type = "cachejpeg_rosetta"
            rosetta_cfg = self.model_config.get("rosetta_config", {})
            is_do_alignment = self.model_config.get(
                "is_do_alignment", rosetta_cfg.get("is_do_alignment", False)
            )
            # A full CacheJPEG-Rosetta wrapper owns the tokenizer loaded with
            # the teacher. Single-model ablations intentionally do not expose
            # it and therefore remain on their native one-tokenizer path.
            llm_tokenizer = (
                getattr(model, "teacher_tokenizer", None) if is_do_alignment else None
            )
        elif self.is_cachejpeg_model:
            model, tokenizer = load_cachejpeg_model(
                self.model_config,
                device=device,
                generation_config=self.generation_config,
            )
            model_type = "cachejpeg"
            llm_tokenizer = None
        elif "rosetta" in self.model_name_normalized:
            model, tokenizer = load_rosetta_model(self.model_config, self.eval_config, device=device, generation_config=self.generation_config)
            # Load LLM tokenizer only if alignment is enabled via eval or model config
            rosetta_cfg = self.model_config.get("rosetta_config", {})
            is_do_alignment = self.model_config.get("is_do_alignment", rosetta_cfg.get("is_do_alignment", False))
            llm_model_path = rosetta_cfg.get("teacher_model")
            llm_tokenizer = None
            if is_do_alignment and llm_model_path:
                try:
                    llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_model_path))
                    if llm_tokenizer.pad_token is None:
                        llm_tokenizer.pad_token = llm_tokenizer.eos_token
                    set_default_chat_template(llm_tokenizer, llm_model_path)
                except Exception as e:
                    print(f"Failed to load LLM tokenizer '{llm_model_path}': {e}")
                    llm_tokenizer = None
            model_type = "rosetta"
        else:
            model, tokenizer = load_hf_model(self.model_config["model_name"], device=device, generation_config=self.generation_config)
            if "Qwen" in self.model_config["model_name"]:
                model_type = "qwen"
            else:
                model_type = "hf"
            llm_tokenizer = None
        
        all_cors = []
        subject_cors = {}
        subcat_cors = defaultdict(list)
        cat_cors = defaultdict(list)
        all_length_stats = []
        cot_logs_all = []
        
        for subject in subjects:
            cors, acc, _, length_stats, cot_logs = self.evaluate_subject(
                subject, model, tokenizer, device, model_type, llm_tokenizer
            )
            if cors is None and self.dataset_name != "longbench":
                continue
            
            all_cors.append(cors)
            subject_cors[subject] = acc
            all_length_stats.extend(length_stats)
            cot_logs_all.extend(cot_logs)
            
            # Organize by subcategories and categories (if applicable)
            if self.dataset_name == "mmlu-redux":
                for subcat in self.dataset_config["subcategories"].get(subject, []):
                    subcat_cors[subcat].append(cors)
                    for cat, subcat_list in self.dataset_config["categories"].items():
                        if subcat in subcat_list:
                            cat_cors[cat].append(cors)
        
        return_dict[rank] = {
            "all_cors": all_cors,
            "subject_cors": subject_cors,
            "subcat_cors": dict(subcat_cors),
            "cat_cors": dict(cat_cors),
            "length_stats": all_length_stats,
            "cot_logs": cot_logs_all
        }
    
    def merge_results(self, results_by_rank: Dict) -> Tuple:
        """
        Merge results from multiple GPUs.
        
        Args:
            results_by_rank: Dictionary of results by rank
            
        Returns:
            Merged results tuple
        """
        all_cors = []
        subject_cors = {}
        subcat_cors = defaultdict(list)
        cat_cors = defaultdict(list)
        all_length_stats = []
        all_cot_logs = []
        
        for result in results_by_rank.values():
            all_cors.extend(result["all_cors"])
            subject_cors.update(result.get("subject_cors", {}))
            all_length_stats.extend(result.get("length_stats", []))
            all_cot_logs.extend(result.get("cot_logs", []))
            
            for k, v in result.get("subcat_cors", {}).items():
                subcat_cors[k].extend(v)
            for k, v in result.get("cat_cors", {}).items():
                cat_cors[k].extend(v)
        
        return all_cors, subject_cors, subcat_cors, cat_cors, all_length_stats, all_cot_logs
    
    def save_results(self, all_cors, subject_cors, subcat_cors, cat_cors, 
                    all_length_stats, all_cot_logs, longbench_subject_metrics: Optional[List[Dict[str, Any]]] = None):
        """
        Save evaluation results.
        
        Args:
            Various result arrays and dictionaries
        """
        # Calculate overall accuracy (skip for LongBench)
        longbench_final_score = None
        if self.dataset_name != "longbench":
            overall_accuracy = np.mean(np.concatenate(all_cors)) if all_cors else 0
        else:
            overall_accuracy = 0
            if longbench_subject_metrics:
                total_weight = sum(int(item.get("num_samples", 0)) for item in longbench_subject_metrics)
                if total_weight > 0:
                    longbench_final_score = sum(
                        float(item.get("score", 0.0)) * int(item.get("num_samples", 0))
                        for item in longbench_subject_metrics
                    ) / total_weight
                else:
                    longbench_final_score = 0.0
                overall_accuracy = longbench_final_score
        
        # Prepare summary
        summary = {
            "model": self.model_config["model_name"],
            "dataset": self.dataset_name,
            "answer_method": self.eval_config["answer_method"],
            "overall_accuracy": overall_accuracy,
            "subjects": subject_cors
        }

        if self.dataset_name == "longbench" and longbench_subject_metrics is not None:
            summary["final_score"] = longbench_final_score if longbench_final_score is not None else overall_accuracy
            summary["subjects"] = {
                item["subject"]: {
                    "metric": item.get("metric"),
                    "score": item.get("score", 0.0),
                    "num_samples": item.get("num_samples", 0),
                    "length_buckets": item.get("length_buckets", {}),
                }
                for item in longbench_subject_metrics
            }
            bucket_weighted_scores: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
            for item in longbench_subject_metrics:
                for bucket_name, bucket_metric in item.get("length_buckets", {}).items():
                    num_samples = int(bucket_metric.get("num_samples", 0))
                    if num_samples <= 0:
                        continue
                    bucket_weighted_scores[bucket_name].append((float(bucket_metric.get("score", 0.0)), num_samples))
            summary["length_buckets"] = {
                bucket_name: {
                    "score": sum(score * count for score, count in values) / sum(count for _, count in values),
                    "num_samples": sum(count for _, count in values),
                }
                for bucket_name, values in bucket_weighted_scores.items()
                if sum(count for _, count in values) > 0
            }
            latency_values = [
                float(item["answer_latency_ms"])
                for item in all_cot_logs
                if item.get("answer_latency_ms") is not None
            ]
            codec_rows = [
                item["cachejpeg_stats"]
                for item in all_cot_logs
                if isinstance(item.get("cachejpeg_stats"), dict)
            ]

            def _mean_field(name: str):
                values = [float(row[name]) for row in codec_rows if row.get(name) is not None]
                return float(np.mean(values)) if values else None

            def _mean_layer_series(name: str):
                series = [row[name] for row in codec_rows if isinstance(row.get(name), list)]
                if not series:
                    return []
                layer_count = max(len(values) for values in series)
                return [
                    float(np.mean([
                        float(values[layer_idx])
                        for values in series
                        if layer_idx < len(values) and values[layer_idx] is not None
                    ]))
                    for layer_idx in range(layer_count)
                ]

            layer_encode_seconds = _mean_layer_series("layer_encode_seconds")
            layer_prefill_seconds = _mean_layer_series("layer_prefill_seconds")
            performance = {
                "num_timed_samples": len(latency_values),
                "end_to_end_total_seconds": float(sum(latency_values) / 1000.0),
                "end_to_end_avg_ms": float(np.mean(latency_values)) if latency_values else None,
                "end_to_end_p50_ms": float(np.percentile(latency_values, 50)) if latency_values else None,
                "end_to_end_p95_ms": float(np.percentile(latency_values, 95)) if latency_values else None,
                "longbench_e_score": float(overall_accuracy),
            }
            # Codec timings are optional.  A non-streaming CacheJPEG run reports
            # whole-cache encode/decode times, while layer-streaming additionally
            # reports per-layer and pipeline timings.  Baseline Rosetta and
            # receiver-only runs have no codec, so do not write misleading null
            # codec fields for those modes.
            if codec_rows:
                encode_seconds = _mean_field("encode_seconds")
                decode_seconds = _mean_field("decode_seconds")
                payload_bytes = _mean_field("payload_bytes")
                if encode_seconds is not None:
                    performance["avg_encode_ms"] = encode_seconds * 1000.0
                if decode_seconds is not None:
                    performance["avg_decode_ms"] = decode_seconds * 1000.0
                if payload_bytes is not None:
                    performance["avg_payload_bytes"] = payload_bytes

                if layer_encode_seconds:
                    performance["avg_layer_encode_ms"] = float(
                        np.mean(layer_encode_seconds) * 1000.0
                    )
                    performance["per_layer_avg_encode_ms"] = [
                        value * 1000.0 for value in layer_encode_seconds
                    ]
                if layer_prefill_seconds:
                    performance["avg_layer_prefill_ms"] = float(
                        np.mean(layer_prefill_seconds) * 1000.0
                    )
                    performance["per_layer_avg_prefill_ms"] = [
                        value * 1000.0 for value in layer_prefill_seconds
                    ]
                pipeline_seconds = _mean_field("pipeline_seconds")
                if pipeline_seconds is not None:
                    performance["avg_pipeline_ms"] = pipeline_seconds * 1000.0

            summary["performance"] = performance
        
        # Add categories and subcategories for MMLU-Redux
        if self.dataset_name == "mmlu-redux":
            summary["categories"] = {
                cat: np.mean(np.concatenate(cors)) if cors else 0
                for cat, cors in cat_cors.items()
            }
            summary["subcategories"] = {
                subcat: np.mean(np.concatenate(cors)) if cors else 0
                for subcat, cors in subcat_cors.items()
            }
        
        # Add length statistics
        if all_length_stats:
            length_summary = self._compute_length_statistics(all_length_stats)
            summary["length_statistics"] = length_summary
        
        # Generate filename
        model_name_for_file = self.model_config["model_name"].split("/")[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary JSON
        summary_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_file}")
        if summary.get("performance") is not None:
            performance_file = self.output_dir / (
                f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_"
                f"{timestamp}_performance.json"
            )
            with open(performance_file, "w") as f:
                json.dump(summary["performance"], f, indent=2)
            perf = summary["performance"]
            print(f"Performance summary saved to {performance_file}")
            perf_parts = [
                f"End-to-end avg: {perf.get('end_to_end_avg_ms') or 0.0:.2f} ms",
                f"LongBench-E score: {perf.get('longbench_e_score') or 0.0:.4f}",
            ]
            if perf.get("avg_encode_ms") is not None:
                perf_parts.append(f"avg encode: {perf['avg_encode_ms']:.2f} ms")
            if perf.get("avg_decode_ms") is not None:
                perf_parts.append(f"avg decode: {perf['avg_decode_ms']:.2f} ms")
            if perf.get("avg_layer_encode_ms") is not None:
                perf_parts.append(f"avg layer encode: {perf['avg_layer_encode_ms']:.2f} ms")
            if perf.get("avg_layer_prefill_ms") is not None:
                perf_parts.append(f"avg layer prefill: {perf['avg_layer_prefill_ms']:.2f} ms")
            print(" | ".join(perf_parts))
        
        # Save detailed length statistics
        if all_length_stats:
            detailed_length_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_length.json"
            with open(detailed_length_file, "w") as f:
                json.dump(all_length_stats, f, indent=2)
            print(f"Detailed length statistics saved to {detailed_length_file}")
        
        # Save CoT logs as CSV or JSONL based on dataset
        if all_cot_logs:
            if self.dataset_name != "longbench":
                cot_csv_file = self.output_dir / f"{model_name_for_file}_{self.dataset_name}_{self.eval_config['answer_method']}_{timestamp}_cot.csv"
                fieldnames = [
                    'subject', 'question_id', 'question', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'true_answer', 'pred', 'is_correct', 'answer_method',
                    'cot_pred', 'cot_input_length', 'cot_gen_length', 'cot_output',
                    'answer_latency_ms',
                    # Extraction diagnostics (mainly for MATH-500)
                    'extraction_method_used', 'ground_truth_normalized', 'extracted_normalized'
                ]
                with open(cot_csv_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in all_cot_logs:
                        writer.writerow(row)
                print(f"CoT outputs saved to {cot_csv_file}")
        
        print(f"\nEvaluation complete!")
        if self.dataset_name != "longbench":
            print(f"Overall accuracy: {overall_accuracy*100:.2f}%")
        elif longbench_subject_metrics is not None:
            print(f"Final score: {overall_accuracy*100:.2f}%")
    
    def _compute_length_statistics(self, length_stats: List[Dict]) -> Dict:
        """
        Compute length statistics summary.
        
        Args:
            length_stats: List of length statistics
            
        Returns:
            Summary dictionary
        """
        def mean_accuracy(stats: List[Dict]) -> Optional[float]:
            values = [
                float(stat["is_correct"])
                for stat in stats
                if stat.get("is_correct") is not None
            ]
            return float(np.mean(values)) if values else None

        if self.dataset_name == "mmlu-redux":
            # Group by subcategory
            subcat_stats = defaultdict(list)
            for stat in length_stats:
                subject = stat['subject']
                for subcat in self.dataset_config["subcategories"].get(subject, []):
                    subcat_stats[subcat].append(stat)
            
            summary = {"subcategories": {}}
            for subcat, stats in subcat_stats.items():
                if stats:
                    summary["subcategories"][subcat] = {
                        "avg_input_length": np.mean([s['input_length'] for s in stats]),
                        "avg_gen_length": np.mean([s['gen_length'] for s in stats]),
                        "avg_length_ratio": np.mean([s['length_ratio'] for s in stats]),
                        "accuracy": mean_accuracy(stats),
                        "total_samples": len(stats)
                    }
        else:
            # Group by subject for MMMLU and LongBench
            subject_stats = defaultdict(list)
            for stat in length_stats:
                subject_stats[stat['subject']].append(stat)
            
            summary = {"subjects": {}}
            for subject, stats in subject_stats.items():
                if stats:
                    summary["subjects"][subject] = {
                        "avg_input_length": np.mean([s['input_length'] for s in stats]),
                        "avg_gen_length": np.mean([s['gen_length'] for s in stats]),
                        "avg_length_ratio": np.mean([s['length_ratio'] for s in stats]),
                        # LongBench records is_correct=None because its F1/EM/
                        # ROUGE score is computed from the prediction file later.
                        "accuracy": mean_accuracy(stats),
                        "total_samples": len(stats)
                    }
        
        return summary
    
    def run(self):
        """Run the evaluation."""
        gpu_ids = self.eval_config["gpu_ids"]
        num_gpus = len(gpu_ids)
        print(f"Using {num_gpus} GPUs: {gpu_ids}")
        # Enable CUDA synchronous errors if requested
        if self.cuda_launch_blocking:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Get subjects for this dataset
        subjects = self.dataset_config["subjects"]
        if self.dataset_name == "longbench" and self.eval_config.get("longbench_e", False):
            subjects = self.dataset_config["subjects_e"]
        if self.dataset_name in ["math-500", "gsm8k", "openbookqa", "gpqa", "ai2-arc", "mmlu-pro"]:
            # Create virtual subject splits to distribute across GPUs
            subjects = self._make_subject_splits(num_gpus)
        
        # Filter subjects if specified in config
        if "subjects" in self.eval_config and self.eval_config["subjects"] is not None:
            requested_subjects = {
                str(subject)[:-2] if str(subject).endswith("_e") else str(subject)
                for subject in self.eval_config["subjects"]
            }
            subjects = [s for s in subjects if s in requested_subjects]
        
        # For LongBench, check if we're evaluating on LongBench-E
        if self.dataset_name == "longbench" and self.eval_config.get("longbench_e", False):
            subjects = [f"{s}_e" for s in subjects]
            self._prepare_longbench_e_test_subset(subjects)
        if(self.dataset_name == "longbench"):
            longbench_subject_metrics = []
            longbench_length_stats = []
            longbench_cot_logs = []
            model_name_for_file = self.model_config["model_name"].split("/")[-1]
            skip_existing = bool(self.eval_config.get("skip_existing_longbench_subjects", True))
            pending_subjects = []
            for subject in subjects:
                is_longbench_e = subject.endswith("_e")
                subject_name = subject[:-2] if is_longbench_e else subject
                pred_dir = self.output_dir / self._longbench_prediction_split_dir(is_longbench_e) / model_name_for_file
                output_file = pred_dir / f"{subject_name}.jsonl"
                if skip_existing and self._has_valid_longbench_output(output_file):
                    print(f"Skipping existing LongBench subject: {subject} -> {output_file}")
                    longbench_subject_metrics.append(self._score_longbench_subject(subject, output_file))
                    continue
                if not skip_existing and output_file.exists():
                    # The evaluator appends samples while running; clear an old
                    # subject file so repeated benchmark commands do not double-count it.
                    output_file.unlink()
                pending_subjects.append(subject)

            if pending_subjects:
                print(f"LongBench pending subjects: {pending_subjects}")
                subject_chunks = [pending_subjects[i::num_gpus] for i in range(num_gpus)]
                run_in_current_process = bool(
                    self.eval_config.get("run_in_current_process", False)
                )
                if run_in_current_process:
                    if num_gpus != 1:
                        raise ValueError(
                            "run_in_current_process requires exactly one configured GPU"
                        )
                    return_dict = {}
                    self.evaluate_on_gpu(
                        0, gpu_ids[0], subject_chunks[0], return_dict
                    )
                else:
                    manager = mp.Manager()
                    return_dict = manager.dict()
                    processes = []

                    for rank, gpu_id in enumerate(gpu_ids):
                        if not subject_chunks[rank]:
                            continue
                        p = mp.Process(
                            target=self.evaluate_on_gpu,
                            args=(rank, gpu_id, subject_chunks[rank], return_dict)
                        )
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()

                    failed_processes = [
                        p for p in processes if p.exitcode != 0
                    ]
                    if failed_processes:
                        exit_codes = [p.exitcode for p in failed_processes]
                        raise RuntimeError(
                            "LongBench evaluation worker process failed; "
                            f"exit_codes={exit_codes}"
                        )

                merged = self.merge_results(return_dict)
                longbench_length_stats = merged[4]
                longbench_cot_logs = merged[5]

                for subject in pending_subjects:
                    is_longbench_e = subject.endswith("_e")
                    subject_name = subject[:-2] if is_longbench_e else subject
                    pred_dir = self.output_dir / self._longbench_prediction_split_dir(is_longbench_e) / model_name_for_file
                    output_file = pred_dir / f"{subject_name}.jsonl"
                    longbench_subject_metrics.append(self._score_longbench_subject(subject, output_file))
            else:
                print("All requested LongBench subjects already have valid jsonl outputs; skipping generation.")

            self.save_results(
                [], {}, {}, {}, longbench_length_stats, longbench_cot_logs,
                longbench_subject_metrics=longbench_subject_metrics,
            )
            return
        else:
        # Distribute subjects across GPUs
            subject_chunks = [subjects[i::num_gpus] for i in range(num_gpus)]
            
            # Launch multi-process evaluation
            manager = mp.Manager()
            return_dict = manager.dict()
            processes = []
            
            for rank, gpu_id in enumerate(gpu_ids):
                p = mp.Process(
                    target=self.evaluate_on_gpu,
                    args=(rank, gpu_id, subject_chunks[rank], return_dict)
                )
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()

            failed_processes = [p for p in processes if p.exitcode != 0]
            if failed_processes:
                exit_codes = [p.exitcode for p in failed_processes]
                raise RuntimeError(
                    "Evaluation worker process failed; "
                    f"exit_codes={exit_codes}"
                )
        # Merge and save results
            results = self.merge_results(return_dict)
            self.save_results(*results)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Unified Evaluation Script')
    parser.add_argument(
        "--config",
        type=str,
        default="eval_recipe/unified_eval.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--cachejpeg-bandwidth-mbps",
        type=float,
        default=None,
        help="Override CacheJPEG transport bandwidth in decimal MB/s.",
    )
    parser.add_argument(
        "--cachejpeg-entropy-backend",
        type=str,
        default=None,
        help=(
            "Override CacheJPEG entropy backend. Supported examples: zlib1, lz4, "
            "zigzag_rle, zigzag_rle_lz4."
        ),
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.cachejpeg_entropy_backend is not None:
        model_config = config.setdefault("model", {})
        if "cachejpeg_config" in model_config:
            model_config["cachejpeg_config"].setdefault("entropy", {})["backend"] = args.cachejpeg_entropy_backend
        elif "cachejpeg_rosetta_config" in model_config:
            codec_config = model_config["cachejpeg_rosetta_config"].setdefault("codec", {})
            codec_config.setdefault("entropy", {})["backend"] = args.cachejpeg_entropy_backend
        else:
            parser.error("--cachejpeg-entropy-backend requires a CacheJPEG model configuration.")

    if args.cachejpeg_bandwidth_mbps is not None:
        if args.cachejpeg_bandwidth_mbps <= 0:
            parser.error("--cachejpeg-bandwidth-mbps must be positive.")
        model_config = config.setdefault("model", {})
        bandwidth_bytes_per_sec = float(args.cachejpeg_bandwidth_mbps) * 1_000_000.0
        if "cachejpeg_config" in model_config:
            model_config["cachejpeg_config"].setdefault("transport", {})[
                "bandwidth_bytes_per_sec"
            ] = bandwidth_bytes_per_sec
        elif "cachejpeg_rosetta_config" in model_config:
            model_config["cachejpeg_rosetta_config"].setdefault("transport", {})[
                "bandwidth_bytes_per_sec"
            ] = bandwidth_bytes_per_sec
        else:
            parser.error("--cachejpeg-bandwidth-mbps requires a CacheJPEG model configuration.")
    
    print("Using config: ", args.config)
    if args.cachejpeg_entropy_backend is not None:
        print("CacheJPEG entropy backend override: ", args.cachejpeg_entropy_backend)
    if args.cachejpeg_bandwidth_mbps is not None:
        print("CacheJPEG transport bandwidth override (MB/s): ", args.cachejpeg_bandwidth_mbps)

    # Preserve the evaluator's historical physical-GPU indexing semantics.
    # GPU selection is controlled by eval.gpu_ids in the YAML.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    # Create and run evaluator
    evaluator = UnifiedEvaluator(config)
    evaluator.run()


if __name__ == "__main__":
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = 64 # you can expand this as needed
    mp.set_start_method("spawn", force=True)
    main()
