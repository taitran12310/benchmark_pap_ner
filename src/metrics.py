"""Strict entity-level Precision / Recall / Micro & Macro F1 (CoNLL-style, word-level)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Entity = Tuple[str, int, int]  # (type, start, end_exclusive)


def bio_to_entities(tags: Sequence[str]) -> Set[Entity]:
    entities: Set[Entity] = set()
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            ent_type = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{ent_type}":
                j += 1
            entities.add((ent_type, i, j))
            i = j
        else:
            i += 1
    return entities


def align_word_predictions(
    y_true_ids: Sequence[int],
    y_pred_ids: Sequence[int],
    id2label: Dict[int, str],
    word_starts: Optional[Sequence[int]] = None,
) -> Tuple[List[str], List[str]]:
    """Keep word-level positions for CoNLL-style entity scoring.

    Prefer ``word_starts==1`` (first subword of each word). Fallback: all
    positions where gold label != -100 (legacy).
    """
    true_tags: List[str] = []
    pred_tags: List[str] = []
    for i, (t, p) in enumerate(zip(y_true_ids, y_pred_ids)):
        if word_starts is not None:
            if int(word_starts[i]) != 1:
                continue
        elif t == -100:
            continue
        true_tags.append(id2label.get(int(t), "O") if t != -100 else "O")
        pred_tags.append(id2label.get(int(p), "O"))
    return true_tags, pred_tags


def score_entities(
    gold: Iterable[Entity],
    pred: Iterable[Entity],
) -> Tuple[int, int, int]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "support": float(tp + fn),
    }


class NerStrictMetric:
    def __init__(self, id2label: Dict[int, str]) -> None:
        self.id2label = id2label
        self.reset()

    def reset(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.per_type = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def update(
        self,
        labels_batch: Sequence[Sequence[int]],
        preds_batch: Sequence[Sequence[int]],
        word_starts_batch: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        for i, (labels, preds) in enumerate(zip(labels_batch, preds_batch)):
            ws = word_starts_batch[i] if word_starts_batch is not None else None
            true_tags, pred_tags = align_word_predictions(
                labels, preds, self.id2label, word_starts=ws
            )
            gold_ents = bio_to_entities(true_tags)
            pred_ents = bio_to_entities(pred_tags)
            tp, fp, fn = score_entities(gold_ents, pred_ents)
            self.tp += tp
            self.fp += fp
            self.fn += fn

            gold_by_type: Dict[str, Set[Entity]] = defaultdict(set)
            pred_by_type: Dict[str, Set[Entity]] = defaultdict(set)
            for e in gold_ents:
                gold_by_type[e[0]].add(e)
            for e in pred_ents:
                pred_by_type[e[0]].add(e)
            types = set(gold_by_type) | set(pred_by_type)
            for t in types:
                t_tp, t_fp, t_fn = score_entities(gold_by_type[t], pred_by_type[t])
                self.per_type[t]["tp"] += t_tp
                self.per_type[t]["fp"] += t_fp
                self.per_type[t]["fn"] += t_fn

    def compute(self) -> Dict[str, Dict[str, float]]:
        overall = prf(self.tp, self.fp, self.fn)
        by_type = {
            t: prf(v["tp"], v["fp"], v["fn"]) for t, v in sorted(self.per_type.items())
        }
        # Macro = unweighted mean over entity types (same as paper Table 7/9)
        if by_type:
            n = float(len(by_type))
            macro = {
                "precision": sum(v["precision"] for v in by_type.values()) / n,
                "recall": sum(v["recall"] for v in by_type.values()) / n,
                "f1": sum(v["f1"] for v in by_type.values()) / n,
            }
        else:
            macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"overall": overall, "macro": macro, "per_entity": by_type}


def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    o = metrics["overall"]
    lines = [
        f"Micro-P: {o['precision']*100:.2f}",
        f"Micro-R: {o['recall']*100:.2f}",
        f"Micro-F1: {o['f1']*100:.2f}",
    ]
    m = metrics.get("macro")
    if m:
        lines.extend(
            [
                f"Macro-P: {m['precision']*100:.2f}",
                f"Macro-R: {m['recall']*100:.2f}",
                f"Macro-F1: {m['f1']*100:.2f}",
            ]
        )
    lines.append("Per-entity F1:")
    for ent, vals in metrics["per_entity"].items():
        lines.append(
            f"  {ent}: P={vals['precision']*100:.2f} "
            f"R={vals['recall']*100:.2f} F1={vals['f1']*100:.2f} "
            f"(support={int(vals['support'])})"
        )
    return "\n".join(lines)
