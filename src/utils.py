"""Utility helpers for reproducibility and environment logging."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Better reproducibility on Colab/Kaggle (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def detect_runtime() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if "COLAB_GPU" in os.environ or Path("/content").exists():
        return "colab"
    return "local"


def looks_like_src(path: Path) -> bool:
    return (path / "train.py").exists() and (path / "config.py").exists()


def resolve_src_dir(explicit: Optional[str] = None) -> Path:
    """Find project `src` on Kaggle input / Colab / local."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    candidates.extend(
        [
            Path("/kaggle/input/pap-ner-src/src"),
            Path("/kaggle/input/pap-ner-src"),
            Path("/content/pap_ner_src"),
            Path("/content/pap-ner-src/src"),
            Path("/content/pap-ner-src"),
            Path("/content/src"),
            Path("/content/drive/MyDrive/PAP_NER/src"),
            Path("/content/drive/MyDrive/pap-ner-src/src"),
            Path("/kaggle/working/src"),
            Path.cwd() / "src",
            Path.cwd(),
        ]
    )

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for cfg in input_root.rglob("config.py"):
            candidates.insert(0, cfg.parent)

    for path in candidates:
        if path.exists() and looks_like_src(path):
            return path.resolve()

    raise FileNotFoundError(
        "Không tìm thấy thư mục src (cần train.py + config.py). "
        "Hãy Add Input dataset pap-ner-src trên Kaggle."
    )


def default_paths() -> Dict[str, Path]:
    runtime = detect_runtime()
    if runtime == "kaggle":
        root = Path("/kaggle/working/pap_ner_benchmark")
    elif runtime == "colab":
        root = Path("/content/pap_ner_benchmark")
    else:
        root = Path.cwd() / "pap_ner_benchmark"
    return {
        "root": root,
        "data": root / "data" / "pap_ner",
        "outputs": root / "outputs",
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "runtime": detect_runtime(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import transformers

        report["transformers"] = transformers.__version__
    except Exception:
        report["transformers"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
    else:
        report["gpu_name"] = None
        report["gpu_vram_gb"] = None
    return report


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def elapsed_str(self) -> str:
        sec = self.elapsed()
        m, s = divmod(sec, 60)
        h, m = divmod(int(m), 60)
        return f"{h:02d}:{m:02d}:{s:05.2f}"


def peak_gpu_memory_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)
