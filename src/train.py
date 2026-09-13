"""Training loop for Transformer(+CRF) on PAP_NER."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from config import TrainConfig
from data import build_tokenizer, load_pap_ner_splits, make_dataloader, prepare_label_maps
from download_data import download_pap_ner
from metrics import NerStrictMetric, format_metrics
from model import TransformerCrfForNer
from schedulers import get_linear_schedule_with_warmup
from utils import (
    Timer,
    environment_report,
    get_device,
    peak_gpu_memory_mb,
    save_json,
    set_seed,
)


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def split_batch_for_model(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """Separate metric-only fields (word_starts) from model inputs."""
    word_starts = batch.get("word_starts")
    model_batch = {k: v for k, v in batch.items() if k != "word_starts"}
    return model_batch, word_starts


def _is_no_decay_param(name: str) -> bool:
    name_l = name.lower()
    return name_l.endswith("bias") or "layernorm" in name_l or ".ln_" in name_l


def build_optimizer(model: TransformerCrfForNer, config: TrainConfig) -> AdamW:
    """Paper Table 4 LRs; no weight decay on bias / LayerNorm."""
    encoder_decay, encoder_nodecay = [], []
    for n, p in model.encoder.named_parameters():
        if not p.requires_grad:
            continue
        (encoder_nodecay if _is_no_decay_param(n) else encoder_decay).append(p)

    classifier_decay, classifier_nodecay = [], []
    for n, p in model.classifier.named_parameters():
        if not p.requires_grad:
            continue
        (classifier_nodecay if _is_no_decay_param(n) else classifier_decay).append(p)

    groups = []
    if encoder_decay:
        groups.append(
            {"params": encoder_decay, "lr": config.learning_rate, "weight_decay": config.weight_decay}
        )
    if encoder_nodecay:
        groups.append(
            {"params": encoder_nodecay, "lr": config.learning_rate, "weight_decay": 0.0}
        )
    if classifier_decay:
        groups.append(
            {
                "params": classifier_decay,
                "lr": config.classifier_learning_rate,
                "weight_decay": config.weight_decay,
            }
        )
    if classifier_nodecay:
        groups.append(
            {
                "params": classifier_nodecay,
                "lr": config.classifier_learning_rate,
                "weight_decay": 0.0,
            }
        )

    if config.use_crf and model.crf is not None:
        crf_decay, crf_nodecay = [], []
        for n, p in model.crf.named_parameters():
            if not p.requires_grad:
                continue
            (crf_nodecay if _is_no_decay_param(n) else crf_decay).append(p)
        if crf_decay:
            groups.append(
                {
                    "params": crf_decay,
                    "lr": config.crf_learning_rate,
                    "weight_decay": config.weight_decay,
                }
            )
        if crf_nodecay:
            groups.append(
                {
                    "params": crf_nodecay,
                    "lr": config.crf_learning_rate,
                    "weight_decay": 0.0,
                }
            )

    return AdamW(groups)


@torch.no_grad()
def evaluate_model(
    model: TransformerCrfForNer,
    dataloader,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict:
    model.eval()
    metric = NerStrictMetric(id2label)
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = move_batch(batch, device)
        model_batch, word_starts = split_batch_for_model(batch)
        out = model(**model_batch, decode=True)
        if "loss" in out:
            total_loss += float(out["loss"].item())
            n_batches += 1
        labels = model_batch["labels"].detach().cpu().tolist()
        preds = out["predictions"].detach().cpu().tolist()
        ws = None if word_starts is None else word_starts.detach().cpu().tolist()
        metric.update(labels, preds, word_starts_batch=ws)
    scores = metric.compute()
    scores["loss"] = total_loss / max(n_batches, 1)
    return scores


def train(config: TrainConfig) -> Dict:
    set_seed(config.seed)
    device = get_device()
    env = environment_report()
    print("[env]", env)

    data_dir = Path(config.data_dir)
    if not (data_dir / "train_data.txt").exists():
        download_pap_ner(data_dir)

    splits = load_pap_ner_splits(
        data_dir,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
        max_test_samples=config.max_test_samples,
    )
    label2id, id2label, label_list = prepare_label_maps(splits)

    model_name = config.resolve_model()
    tokenizer = build_tokenizer(model_name)
    train_loader = make_dataloader(
        splits["train"],
        tokenizer,
        label2id,
        max_length=config.max_length,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    dev_loader = make_dataloader(
        splits["dev"],
        tokenizer,
        label2id,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = make_dataloader(
        splits["test"],
        tokenizer,
        label2id,
        max_length=config.max_length,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = TransformerCrfForNer(
        model_name_or_path=model_name,
        num_labels=len(label_list),
        use_crf=config.use_crf,
    ).to(device)

    optimizer = build_optimizer(model, config)
    total_steps = (
        len(train_loader) * config.num_epochs // max(config.gradient_accumulation_steps, 1)
    )
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    use_amp = bool(config.fp16 and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "config": config.to_dict(),
            "model_name": model_name,
            "label_list": label_list,
            "num_parameters": model.num_parameters(),
            "environment": env,
            "protocol": "paper_like_table4",
        },
        run_dir / "run_meta.json",
    )

    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    train_timer = Timer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    global_step = 0
    stopped_early = False
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_batch(batch, device)
            model_batch, _word_starts = split_batch_for_model(batch)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(**model_batch, decode=False)
                loss = out["loss"] / config.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if step % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                # GradScaler may skip optimizer.step() on inf/nan; only then skip scheduler
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += float(out["loss"].item())
            if step % config.logging_steps == 0:
                pbar.set_postfix(loss=running_loss / step)

        dev_scores = evaluate_model(model, dev_loader, id2label, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "dev": dev_scores,
        }
        history.append(epoch_row)
        print(f"\n[epoch {epoch}] train_loss={epoch_row['train_loss']:.4f}")
        print(format_metrics(dev_scores))

        dev_f1 = dev_scores["overall"]["f1"]
        improved = dev_f1 > best_f1 + config.early_stopping_min_delta
        if improved:
            best_f1 = dev_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "config": config.to_dict(),
                    "model_name": model_name,
                    "best_epoch": best_epoch,
                },
                run_dir / "best.pt",
            )
            print(f"[checkpoint] new best Micro-F1={best_f1*100:.2f} -> {run_dir/'best.pt'}")
        else:
            epochs_no_improve += 1
            print(
                f"[early-stop] no improve {epochs_no_improve}/"
                f"{config.early_stopping_patience} "
                f"(best={best_f1*100:.2f} @ ep{best_epoch}, "
                f"min_delta={config.early_stopping_min_delta})"
            )
            if (
                config.early_stopping_patience > 0
                and epochs_no_improve >= config.early_stopping_patience
            ):
                stopped_early = True
                print(
                    f"[early-stop] stop at epoch {epoch} "
                    f"(reason=patience; best_epoch={best_epoch})"
                )
                break

    train_time = train_timer.elapsed()
    # Load best and evaluate on test
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_scores = evaluate_model(model, test_loader, id2label, device)
    print("\n=== TEST (best checkpoint) ===")
    print(format_metrics(test_scores))

    result = {
        "model_name": model_name,
        "use_crf": config.use_crf,
        "best_dev_micro_f1": best_f1,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "epochs_ran": len(history),
        "test": test_scores,
        "history": history,
        "train_time_sec": train_time,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "num_parameters": model.num_parameters(),
        "paper_baseline_micro_f1": 0.9795,
        "environment": env,
    }
    save_json(result, run_dir / "results.json")
    tokenizer.save_pretrained(run_dir / "tokenizer")
    return result


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train PLM(+CRF) on PAP_NER")
    p.add_argument("--model_key", type=str, default="phobert")
    p.add_argument("--model_name_or_path", type=str, default="")
    p.add_argument("--use_crf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--data_dir", type=str, default="data/pap_ner")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--classifier_learning_rate", type=float, default=1e-3)
    p.add_argument("--crf_learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--max_test_samples", type=int, default=-1)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
