"""Qualitative error analysis helpers (Research_EN.md §15)."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set

from metrics import Entity, align_word_predictions, bio_to_entities


def filter_ents_by_type(ents: Iterable[Entity], ent_type: str) -> Set[Entity]:
    return {e for e in ents if e[0] == ent_type}


def classify_errors(
    gold_ents: set,
    pred_ents: set,
) -> Dict[str, List]:
    """Bucket errors into boundary / type / missing / spurious."""
    exact = gold_ents & pred_ents
    gold_spans = {(s, e): t for t, s, e in gold_ents}
    pred_spans = {(s, e): t for t, s, e in pred_ents}

    type_errors = []
    for span, g_type in gold_spans.items():
        if span in pred_spans and pred_spans[span] != g_type:
            type_errors.append(
                {"span": span, "gold": g_type, "pred": pred_spans[span]}
            )

    type_spans = {tuple(t["span"]) for t in type_errors}

    boundary_errors = []
    for g_type, gs, ge in gold_ents - exact:
        if (gs, ge) in type_spans:
            continue
        for p_type, ps, pe in pred_ents - exact:
            if (ps, pe) in type_spans:
                continue
            overlap = max(0, min(ge, pe) - max(gs, ps))
            if overlap > 0 and (gs, ge) != (ps, pe):
                boundary_errors.append(
                    {
                        "gold": (g_type, gs, ge),
                        "pred": (p_type, ps, pe),
                        "overlap": overlap,
                    }
                )

    bound_gold = {tuple(b["gold"]) for b in boundary_errors}
    bound_pred = {tuple(b["pred"]) for b in boundary_errors}

    missing = sorted(
        e
        for e in (gold_ents - pred_ents)
        if (e[1], e[2]) not in type_spans and e not in bound_gold
    )
    spurious = sorted(
        e
        for e in (pred_ents - gold_ents)
        if (e[1], e[2]) not in type_spans and e not in bound_pred
    )
    return {
        "exact_match": sorted(exact),
        "type_errors": type_errors,
        "boundary_errors": boundary_errors,
        "missing": missing,
        "spurious": spurious,
    }


def focus_error_buckets(buckets: Dict[str, List], focus_type: str) -> Dict[str, List]:
    """Keep only error items that involve ``focus_type`` (e.g. ĐT)."""

    def _ent_ok(ent) -> bool:
        return ent[0] == focus_type

    def _type_ok(item: Dict) -> bool:
        return item["gold"] == focus_type or item["pred"] == focus_type

    def _bound_ok(item: Dict) -> bool:
        return item["gold"][0] == focus_type or item["pred"][0] == focus_type

    return {
        "exact_match": [e for e in buckets["exact_match"] if _ent_ok(e)],
        "type_errors": [t for t in buckets["type_errors"] if _type_ok(t)],
        "boundary_errors": [b for b in buckets["boundary_errors"] if _bound_ok(b)],
        "missing": [e for e in buckets["missing"] if _ent_ok(e)],
        "spurious": [e for e in buckets["spurious"] if _ent_ok(e)],
    }


def analyze_sentence(
    tokens: Sequence[str],
    true_ids: Sequence[int],
    pred_ids: Sequence[int],
    id2label: Dict[int, str],
    word_starts: Optional[Sequence[int]] = None,
    focus_type: Optional[str] = None,
    truncated: bool = False,
) -> Dict:
    true_tags, pred_tags = align_word_predictions(
        true_ids, pred_ids, id2label, word_starts=word_starts
    )
    if len(tokens) >= len(true_tags):
        word_tokens = list(tokens[: len(true_tags)])
    else:
        word_tokens = list(tokens)

    gold_ents = bio_to_entities(true_tags)
    pred_ents = bio_to_entities(pred_tags)
    buckets = classify_errors(gold_ents, pred_ents)
    if focus_type:
        buckets = focus_error_buckets(buckets, focus_type)

    return {
        "text": " ".join(word_tokens),
        "tokens": word_tokens,
        "true_tags": true_tags,
        "pred_tags": pred_tags,
        "gold_entities": sorted(gold_ents),
        "pred_entities": sorted(pred_ents),
        "errors": buckets,
        "truncated": truncated,
        "focus_type": focus_type,
    }


def summarize_error_types(analyses: List[Dict]) -> Dict[str, int]:
    counter = Counter()
    for a in analyses:
        err = a["errors"]
        counter["exact_match"] += len(err["exact_match"])
        counter["type_errors"] += len(err["type_errors"])
        counter["boundary_errors"] += len(err["boundary_errors"])
        counter["missing"] += len(err["missing"])
        counter["spurious"] += len(err["spurious"])
        if a.get("truncated"):
            counter["sentences_truncated"] += 1
    return dict(counter)


def pick_examples_by_category(
    analyses: List[Dict],
    category: str,
    limit: int = 5,
) -> List[Dict]:
    picked = []
    for a in analyses:
        if a["errors"].get(category):
            picked.append(a)
        if len(picked) >= limit:
            break
    return picked


def long_entity_errors(
    analyses: List[Dict],
    min_tokens: int = 8,
) -> List[Dict]:
    """Research §15.3 — focus on long administrative spans."""
    out = []
    for a in analyses:
        long_missing = [
            e for e in a["errors"]["missing"] if (e[2] - e[1]) >= min_tokens
        ]
        if long_missing:
            out.append({**a, "long_missing": long_missing})
    return out


def span_length_histogram(
    analyses: List[Dict],
    category: str = "missing",
) -> Dict[str, int]:
    """Bucket entity token lengths for a given error category."""
    bins = Counter()
    for a in analyses:
        for e in a["errors"].get(category, []):
            n = e[2] - e[1]
            if n <= 2:
                bins["1-2"] += 1
            elif n <= 5:
                bins["3-5"] += 1
            elif n <= 10:
                bins["6-10"] += 1
            else:
                bins["11+"] += 1
    return dict(bins)
