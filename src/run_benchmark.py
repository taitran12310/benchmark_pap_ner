"""Run paper-like model-swap benchmark (CRF fixed; only change PLM).

Protocol follows La et al. PAP_NER Table 4:
  full data, AdamW, encoder LR 2e-5, CRF LR 1e-3, batch 32 (16 for xlmr),
  max 20 epochs, early stopping patience 3, max_len 256, warmup 0.1, FP16.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import TrainConfig, paper_like_batch_size, paper_model_swap_jobs
from train import train
from utils import default_paths, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-like PLM swap on PAP_NER (CRF fixed)"
    )
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-3)
    parser.add_argument("--crf_learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_test_samples", type=int, default=-1)
    parser.add_argument(
        "--models",
        type=str,
        default="phobert,phobert_v2,xlmr",
        help="Comma-separated model keys (CRF on for all)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds (paper reports 3 seeds for mean±std)",
    )
    parser.add_argument(
        "--with_ablation",
        action="store_true",
        help="Also train Softmax (no CRF) variants",
    )
    args = parser.parse_args()

    paths = default_paths()
    data_dir = args.data_dir or str(paths["data"])
    output_dir = args.output_dir or str(paths["outputs"])

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = paper_model_swap_jobs(models=model_keys, seeds=seeds)
    if args.with_ablation:
        extra = []
        for job in jobs:
            soft = dict(job)
            soft["use_crf"] = False
            extra.append(soft)
        jobs.extend(extra)

    summary = []
    for job in jobs:
        bs = paper_like_batch_size(job["model_key"], args.batch_size)
        cfg = TrainConfig(
            model_key=job["model_key"],
            use_crf=job["use_crf"],
            data_dir=data_dir,
            output_dir=output_dir,
            num_epochs=args.num_epochs,
            batch_size=bs,
            learning_rate=args.learning_rate,
            classifier_learning_rate=args.classifier_learning_rate,
            crf_learning_rate=args.crf_learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_test_samples=args.max_test_samples,
            seed=job["seed"],
        )
        print("\n" + "=" * 72)
        print(
            f"Running {cfg.model_key} | CRF={cfg.use_crf} | seed={cfg.seed} | "
            f"bs={cfg.batch_size} | epochs<={cfg.num_epochs}"
        )
        print("=" * 72)
        result = train(cfg)
        summary.append(
            {
                "model_key": cfg.model_key,
                "use_crf": cfg.use_crf,
                "seed": cfg.seed,
                "batch_size": cfg.batch_size,
                "best_epoch": result.get("best_epoch"),
                "stopped_early": result.get("stopped_early"),
                "test_micro_f1": result["test"]["overall"]["f1"],
                "test_macro_f1": (result["test"].get("macro") or {}).get("f1"),
                "test_precision": result["test"]["overall"]["precision"],
                "test_recall": result["test"]["overall"]["recall"],
                "per_entity": result["test"]["per_entity"],
                "max_length": cfg.max_length,
                "train_time_sec": result["train_time_sec"],
                "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
                "num_parameters": result["num_parameters"],
                "run_dir": str(cfg.run_dir()),
            }
        )

    out = Path(output_dir) / "benchmark_summary.json"
    save_json(summary, out)
    print(f"\nBenchmark summary -> {out}")
    for row in summary:
        print(
            f"- {row['model_key']} CRF={row['use_crf']} seed={row['seed']}: "
            f"Micro-F1={row['test_micro_f1']*100:.2f} "
            f"(best_ep={row['best_epoch']})"
        )


if __name__ == "__main__":
    main()
