"""Evaluate a saved checkpoint on PAP_NER test/dev split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import TrainConfig
from data import build_tokenizer, load_pap_ner_splits, make_dataloader
from metrics import format_metrics
from model import TransformerCrfForNer
from train import evaluate_model
from utils import get_device, save_json, set_seed


def load_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in TrainConfig.__dataclass_fields__})
    model = TransformerCrfForNer(
        model_name_or_path=ckpt["model_name"],
        num_labels=len(ckpt["label2id"]),
        use_crf=cfg.use_crf,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    tok_dir = ckpt_path.parent / "tokenizer"
    if tok_dir.exists():
        tokenizer = build_tokenizer(str(tok_dir))
    else:
        tokenizer = build_tokenizer(ckpt["model_name"])
    return model, tokenizer, ckpt, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/pap_ner")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    args = parser.parse_args()

    device = get_device()
    model, tokenizer, ckpt, cfg = load_checkpoint(Path(args.checkpoint), device)
    set_seed(cfg.seed)

    limits = {"train": -1, "dev": -1, "test": -1}
    limits[args.split] = args.max_samples
    splits = load_pap_ner_splits(
        args.data_dir,
        max_train_samples=limits["train"],
        max_eval_samples=limits["dev"],
        max_test_samples=limits["test"],
    )
    loader = make_dataloader(
        splits[args.split],
        tokenizer,
        ckpt["label2id"],
        max_length=cfg.max_length,
        batch_size=args.batch_size,
        shuffle=False,
    )
    # id2label in ckpt may have int keys serialized as str
    id2label = {int(k): v for k, v in ckpt["id2label"].items()}
    scores = evaluate_model(model, loader, id2label, device)
    print(format_metrics(scores))
    out = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    save_json(scores, out)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
