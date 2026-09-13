"""Experiment configuration for PAP_NER PLM benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


ENTITY_TYPES = ["CQ", "VBPL", "ĐT", "NG", "SL"]

LABEL_LIST = [
    "O",
    "B-CQ",
    "I-CQ",
    "B-VBPL",
    "I-VBPL",
    "B-ĐT",
    "I-ĐT",
    "B-NG",
    "I-NG",
    "B-SL",
    "I-SL",
]

# Baseline + recent PLM candidates (research §9 / §12)
MODEL_REGISTRY: Dict[str, str] = {
    "phobert": "vinai/phobert-base",
    "phobert_v2": "vinai/phobert-base-v2",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
    # Newer VN encoder (VinAI/Qualcomm); raw-text capable, max context 2048.
    # On PAP_NER we still align word→subword (encode_example) for fair word-level eval.
    "bamibert": "Qualcomm-AI-Research/BamiBERT",
}

ZENODO_RECORD_ID = "18044019"
ZENODO_FILES = {
    "train": "train_data.txt",
    "dev": "dev_data.txt",
    "test": "test_data.txt",
}


@dataclass
class TrainConfig:
    """Defaults align with La et al. PAP_NER Table 4 (paper protocol)."""

    model_key: str = "phobert"
    model_name_or_path: str = ""
    use_crf: bool = True

    data_dir: str = "data/pap_ner"
    output_dir: str = "outputs"

    max_length: int = 256
    batch_size: int = 32
    eval_batch_size: int = 64
    num_epochs: int = 20
    learning_rate: float = 2e-5
    classifier_learning_rate: float = 1e-3
    crf_learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    seed: int = 42
    num_workers: int = 2
    logging_steps: int = 100
    save_total_limit: int = 2

    # -1 = full split (paper). Positive = subset for smoke tests.
    max_train_samples: int = -1
    max_eval_samples: int = -1
    max_test_samples: int = -1

    fp16: bool = True
    gradient_accumulation_steps: int = 1

    def resolve_model(self) -> str:
        if self.model_name_or_path:
            return self.model_name_or_path
        if self.model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model_key={self.model_key}. "
                f"Choose from {list(MODEL_REGISTRY)}"
            )
        return MODEL_REGISTRY[self.model_key]

    def run_dir(self) -> Path:
        tag = "crf" if self.use_crf else "softmax"
        return Path(self.output_dir) / f"{self.model_key}_{tag}_seed{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def paper_like_batch_size(model_key: str, default_batch: int = 32) -> int:
    """XLM-R is ~2× PhoBERT; paper used batch 16 for XLM-R (Softmax).

    Keep 16 for heavier encoders on T4 / 16GB cards to avoid OOM while still
    matching the paper's effective batch for that family.
    """
    if model_key in {"xlmr", "mdeberta", "bamibert"}:
        return min(default_batch, 16)
    return default_batch


def paper_like_max_length(model_key: str, default: int = 256) -> int:
    """Default max_length for fair swap vs PhoBERT Table 4 (256).

    BamiBERT supports up to 2048; keep 256 unless caller overrides (e.g. 512)
    when probing long ĐT/VBPL spans.
    """
    _ = model_key
    return default


def paper_model_swap_jobs(
    models: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[dict]:
    """Fair comparison: keep CRF + paper HPs; only swap PLM (+ optional multi-seed)."""
    models = models or ["phobert", "phobert_v2", "xlmr", "bamibert"]
    seeds = seeds or [42]
    return [
        {"model_key": key, "use_crf": True, "seed": seed}
        for seed in seeds
        for key in models
    ]


def default_label_maps() -> tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def experiment_matrix() -> List[dict]:
    """Main experiments in Research_EN.md §12 + ablation §13."""
    rows = []
    for key in ["phobert", "phobert_v2", "xlmr", "mdeberta", "bamibert"]:
        for use_crf in [True, False]:
            rows.append({"model_key": key, "use_crf": use_crf})
    return rows
