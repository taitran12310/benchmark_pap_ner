"""PAP_NER CoNLL loading, tokenization alignment, and DataLoader builders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from config import LABEL_LIST, ZENODO_FILES, default_label_maps


@dataclass
class NerExample:
    tokens: List[str]
    labels: List[str]


def read_conll_file(path: Path, max_samples: int = -1) -> List[NerExample]:
    """Parse space-separated CoNLL: Word POS Chunk NER [extra...]."""
    examples: List[NerExample] = []
    tokens: List[str] = []
    labels: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                if tokens:
                    examples.append(NerExample(tokens=tokens, labels=labels))
                    tokens, labels = [], []
                    if max_samples > 0 and len(examples) >= max_samples:
                        break
                continue

            parts = line.split()
            if len(parts) < 4:
                # Robust fallback: last column as NER if short line
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    labels.append(parts[-1])
                continue

            tokens.append(parts[0])
            labels.append(parts[3])

    if tokens and (max_samples <= 0 or len(examples) < max_samples):
        examples.append(NerExample(tokens=tokens, labels=labels))
    return examples


def load_pap_ner_splits(
    data_dir: str | Path,
    max_train_samples: int = -1,
    max_eval_samples: int = -1,
    max_test_samples: int = -1,
) -> Dict[str, List[NerExample]]:
    data_dir = Path(data_dir)
    limits = {
        "train": max_train_samples,
        "dev": max_eval_samples,
        "test": max_test_samples,
    }
    splits: Dict[str, List[NerExample]] = {}
    for split, filename in ZENODO_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run download_data.py or notebook download cell first."
            )
        splits[split] = read_conll_file(path, max_samples=limits[split])
        print(f"[data] {split}: {len(splits[split]):,} sentences from {path.name}")
    return splits


def build_label_list_from_data(splits: Dict[str, List[NerExample]]) -> List[str]:
    found = set()
    for examples in splits.values():
        for ex in examples:
            found.update(ex.labels)
    # Keep canonical order, append unexpected labels at the end
    ordered = [lab for lab in LABEL_LIST if lab in found]
    extras = sorted(found - set(LABEL_LIST))
    return ordered + extras


class PapNerDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[NerExample],
        tokenizer: PreTrainedTokenizerBase,
        label2id: Dict[str, int],
        max_length: int = 256,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return encode_example(
            tokens=ex.tokens,
            labels=ex.labels,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.max_length,
        )


def bio_subword_label_ids(label: str, label2id: Dict[str, int]) -> tuple[int, int]:
    """Map a word-level BIO tag to (first_subword_id, continuation_id).

    Continuation: B-X → I-X; I-X / O stay the same. Avoids -100 holes inside words
    so linear-chain CRF masks stay contiguous (paper / pytorch-crf assumption).
    """
    first = label2id.get(label, label2id["O"])
    if label.startswith("B-"):
        cont_lab = "I-" + label[2:]
        cont = label2id.get(cont_lab, first)
    else:
        cont = first
    return first, cont


def encode_example(
    tokens: List[str],
    labels: List[str],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Word→subword encoding with B→I fill on continuation pieces (CRF-safe).

    [CLS]/[SEP] stay -100; content subwords all receive BIO ids (no mid-word holes).
    ``word_starts`` marks the first subword of each word (for word-level eval).
    """
    input_ids: List[int] = []
    label_ids: List[int] = []
    attention_mask: List[int] = []
    word_starts: List[int] = []

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    if cls_id is not None:
        input_ids.append(cls_id)
        label_ids.append(-100)
        attention_mask.append(1)
        word_starts.append(0)

    # Reserve room for [SEP]
    budget = max_length - (1 if sep_id is not None else 0)

    for word_idx, word in enumerate(tokens):
        if len(input_ids) >= budget:
            break
        word_pieces = tokenizer.tokenize(word)
        if not word_pieces:
            word_pieces = [tokenizer.unk_token] if tokenizer.unk_token else []
        if not word_pieces:
            continue

        piece_ids = tokenizer.convert_tokens_to_ids(word_pieces)
        # Truncate mid-word if needed
        remain = budget - len(input_ids)
        if remain <= 0:
            break
        piece_ids = piece_ids[:remain]

        lab = labels[word_idx] if word_idx < len(labels) else "O"
        first_id, cont_id = bio_subword_label_ids(lab, label2id)
        for i, pid in enumerate(piece_ids):
            input_ids.append(int(pid))
            attention_mask.append(1)
            label_ids.append(first_id if i == 0 else cont_id)
            word_starts.append(1 if i == 0 else 0)

    if sep_id is not None and len(input_ids) < max_length:
        input_ids.append(sep_id)
        label_ids.append(-100)
        attention_mask.append(1)
        word_starts.append(0)

    # Hard truncate safety
    input_ids = input_ids[:max_length]
    label_ids = label_ids[:max_length]
    attention_mask = attention_mask[:max_length]
    word_starts = word_starts[:max_length]

    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(label_ids, dtype=torch.long),
        "word_starts": torch.tensor(word_starts, dtype=torch.long),
    }
    return out


def collate_batch(
    features: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    max_len = max(f["input_ids"].size(0) for f in features)
    batch: Dict[str, List[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "word_starts": [],
    }

    for f in features:
        pad_len = max_len - f["input_ids"].size(0)
        batch["input_ids"].append(
            torch.nn.functional.pad(f["input_ids"], (0, pad_len), value=pad_token_id)
        )
        batch["attention_mask"].append(
            torch.nn.functional.pad(f["attention_mask"], (0, pad_len), value=0)
        )
        batch["labels"].append(
            torch.nn.functional.pad(f["labels"], (0, pad_len), value=-100)
        )
        batch["word_starts"].append(
            torch.nn.functional.pad(f["word_starts"], (0, pad_len), value=0)
        )

    return {k: torch.stack(v, dim=0) for k, v in batch.items()}


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load tokenizer; tolerate models that need trust_remote_code (e.g. some hubs)."""
    kwargs = {"use_fast": True}
    # PhoBERT often has no usable fast tokenizer; prefer slow if fast unavailable
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        # Probe word_ids support
        enc = tokenizer(["test"], is_split_into_words=True)
        _ = enc.word_ids()
        return tokenizer
    except Exception:
        try:
            return AutoTokenizer.from_pretrained(
                model_name, use_fast=False, trust_remote_code=True
            )
        except Exception:
            return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def make_dataloader(
    examples: Sequence[NerExample],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    max_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = PapNerDataset(examples, tokenizer, label2id, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # Rare for RoBERTa-like models; fall back to eos
        pad_id = tokenizer.eos_token_id or 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda feats: collate_batch(feats, pad_token_id=pad_id),
    )


def prepare_label_maps(
    splits: Optional[Dict[str, List[NerExample]]] = None,
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    if splits is None:
        label2id, id2label = default_label_maps()
        return label2id, id2label, LABEL_LIST
    labels = build_label_list_from_data(splits)
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, labels
