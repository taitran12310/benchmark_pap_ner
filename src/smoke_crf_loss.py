"""Quick regression checks for CRF-safe labeling, word-level metric, NLL>=0."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_word_level_metric() -> None:
    from config import default_label_maps
    from metrics import NerStrictMetric, align_word_predictions, bio_to_entities

    label2id, id2label = default_label_maps()
    b = label2id["B-CQ"]
    i = label2id["I-CQ"]
    o = label2id["O"]
    labels = [-100, b, i, o, -100]
    preds = [-100, b, i, o, -100]
    starts = [0, 1, 0, 1, 0]
    true_tags, pred_tags = align_word_predictions(
        labels, preds, id2label, word_starts=starts
    )
    assert true_tags == ["B-CQ", "O"]
    assert pred_tags == ["B-CQ", "O"]
    ents = bio_to_entities(true_tags)
    assert ents == {("CQ", 0, 1)}

    metric = NerStrictMetric(id2label)
    metric.update([labels], [preds], word_starts_batch=[starts])
    scores = metric.compute()
    assert abs(scores["overall"]["f1"] - 1.0) < 1e-9
    print("[ok] word-level metric uses first subword only")


def test_bio_fill_and_word_starts() -> None:
    import torch  # noqa: F401
    from config import default_label_maps
    from data import bio_subword_label_ids, encode_example

    class _FakeTok:
        cls_token_id = 0
        sep_token_id = 2
        pad_token_id = 1
        unk_token = "[UNK]"

        def tokenize(self, word: str):
            if len(word) > 4:
                mid = max(1, len(word) // 2)
                return [word[:mid], word[mid:]]
            return [word]

        def convert_tokens_to_ids(self, pieces):
            return [10 + i for i, _ in enumerate(pieces)]

    label2id, _ = default_label_maps()
    tok = _FakeTok()
    tokens = ["HaNoi", "ngay", "01"]
    labels = ["B-CQ", "O", "B-NG"]
    feat = encode_example(tokens, labels, tok, label2id, max_length=32)
    labs = feat["labels"].tolist()
    starts = feat["word_starts"].tolist()
    assert starts[0] == 0
    content = labs[1:]
    while content and content[-1] == -100:
        content.pop()
    assert -100 not in content, f"mid-sequence holes remain: {labs}"
    b_cq, i_cq = bio_subword_label_ids("B-CQ", label2id)
    assert labs[1] == b_cq and labs[2] == i_cq
    assert starts[1] == 1 and starts[2] == 0
    print("[ok] encode_example B→I fill + word_starts")


def test_crf_nll_nonnegative() -> None:
    import torch
    from config import LABEL_LIST
    from crf import CRF

    torch.manual_seed(0)
    crf = CRF(num_tags=len(LABEL_LIST), batch_first=True)
    B, T, C = 2, 5, len(LABEL_LIST)
    emissions = torch.randn(B, T, C)
    tags = torch.randint(0, C, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -1] = False
    llh = crf(emissions, tags, mask=mask, reduction="mean")
    nll = -llh
    assert nll.item() >= 0, f"expected NLL>=0, got {nll.item()}"
    print(f"[ok] CRF NLL={nll.item():.4f} >= 0")


def test_model_forward_loss_nonnegative(data_dir: Path) -> None:
    import torch
    from data import build_tokenizer, load_pap_ner_splits, make_dataloader, prepare_label_maps
    from model import TransformerCrfForNer
    from train import split_batch_for_model

    if not (data_dir / "train_data.txt").exists():
        print("[skip] model forward (no local PAP_NER data)")
        return

    splits = load_pap_ner_splits(
        data_dir, max_train_samples=8, max_eval_samples=4, max_test_samples=4
    )
    label2id, id2label, label_list = prepare_label_maps(splits)
    tokenizer = build_tokenizer("vinai/phobert-base")
    loader = make_dataloader(
        splits["train"],
        tokenizer,
        label2id,
        max_length=128,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cpu")
    model = TransformerCrfForNer(
        "vinai/phobert-base", num_labels=len(label_list), use_crf=True
    ).to(device)
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    assert "word_starts" in batch
    model_batch, _ws = split_batch_for_model(batch)
    with torch.no_grad():
        out = model(**model_batch, decode=False)
    loss = float(out["loss"].item())
    assert loss >= 0, f"expected model NLL>=0, got {loss}"
    assert "predictions" not in out
    out_dec = model(**model_batch, decode=True)
    assert "predictions" in out_dec
    print(
        f"[ok] PhoBERT+CRF batch loss={loss:.4f} >= 0; decode flag OK; word_starts present"
    )


def main() -> None:
    test_word_level_metric()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[skip] torch-dependent checks (install torch on Kaggle/runtime)")
        print("[DONE] smoke_crf_loss (partial)")
        return

    test_bio_fill_and_word_starts()
    test_crf_nll_nonnegative()
    data_dir = ROOT / "data" / "pap_ner"
    try:
        test_model_forward_loss_nonnegative(data_dir)
    except Exception as e:
        print(
            f"[warn] model forward check failed (often offline HF): "
            f"{type(e).__name__}: {e}"
        )
    print("[DONE] smoke_crf_loss")


if __name__ == "__main__":
    main()
