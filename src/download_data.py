"""Locate / copy / optionally download PAP_NER data."""

from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from config import ZENODO_FILES, ZENODO_RECORD_ID

USER_AGENT = "pap-ner-benchmark/0.1 (+kaggle; research)"
REQUIRED_FILES = list(ZENODO_FILES.values())


def zenodo_urls(filename: str) -> list[str]:
    return [
        f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/{filename}?download=1",
        f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{filename}/content",
    ]


def has_complete_data(folder: Path) -> bool:
    if not folder.exists():
        return False
    return all((folder / name).exists() and (folder / name).stat().st_size > 0 for name in REQUIRED_FILES)


def candidate_data_dirs(extra: Optional[Iterable[Path]] = None) -> List[Path]:
    dirs: List[Path] = []
    if extra:
        dirs.extend(list(extra))

    # Bundled with source
    dirs.append(Path(__file__).resolve().parent / "data" / "pap_ner")

    # Copied working tree
    dirs.append(Path("/kaggle/working/src/data/pap_ner"))
    dirs.append(Path("/kaggle/working/pap_ner_benchmark/data/pap_ner"))

    # Raw Kaggle input search
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for train in input_root.rglob("train_data.txt"):
            dirs.append(train.parent)

    dirs.append(Path.cwd() / "data" / "pap_ner")
    # dedupe preserve order
    seen = set()
    out = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def find_bundled_data(extra: Optional[Iterable[Path]] = None) -> Optional[Path]:
    for d in candidate_data_dirs(extra):
        if has_complete_data(d):
            return d.resolve()
    return None


def copy_data(src: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        s = src / name
        d = dest / name
        if d.exists() and d.stat().st_size == s.stat().st_size:
            print(f"[skip] {name} already in {dest}")
            continue
        print(f"[copy] {s} -> {d}")
        shutil.copy2(s, d)
    return dest


def download_file(
    filename: str,
    dest: Path,
    chunk_size: int = 1 << 20,
    max_retries: int = 6,
    timeout: int = 180,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return

    urls = zenodo_urls(filename)
    last_err: Exception | None = None
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        url = urls[(attempt - 1) % len(urls)]
        try:
            print(f"[download] try {attempt}/{max_retries}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
                total = resp.headers.get("Content-Length")
                total_i = int(total) if total else None
                done = 0
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total_i:
                        pct = 100.0 * done / total_i
                        print(f"\r  {done:,}/{total_i:,} ({pct:.1f}%)", end="", flush=True)
                print()
            if tmp.stat().st_size <= 0:
                raise RuntimeError("downloaded empty file")
            tmp.replace(dest)
            print(f"[ok] saved -> {dest} ({dest.stat().st_size:,} bytes)")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[warn] failed: {type(exc).__name__}: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Cannot download {filename}") from last_err


def download_from_zenodo(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        download_file(filename, data_dir / filename)
    return data_dir


def ensure_pap_ner_data(
    data_dir: str | Path,
    allow_download: bool = True,
) -> Path:
    """
    Prefer bundled/local data. Optionally fall back to Zenodo.
    """
    data_dir = Path(data_dir)
    if has_complete_data(data_dir):
        print(f"[data] using existing {data_dir}")
        return data_dir.resolve()

    bundled = find_bundled_data()
    if bundled is not None:
        print(f"[data] found bundled dataset at {bundled}")
        return copy_data(bundled, data_dir)

    if not allow_download:
        raise FileNotFoundError(
            "Không thấy PAP_NER local. Hãy đặt 3 file vào src/data/pap_ner/ rồi pack zip.\n"
            "Links:\n"
            f"  https://zenodo.org/records/{ZENODO_RECORD_ID}/files/train_data.txt?download=1\n"
            f"  https://zenodo.org/records/{ZENODO_RECORD_ID}/files/dev_data.txt?download=1\n"
            f"  https://zenodo.org/records/{ZENODO_RECORD_ID}/files/test_data.txt?download=1"
        )

    print("[data] no local bundle — trying Zenodo ...")
    return download_from_zenodo(data_dir)


# Backward-compatible alias
def download_pap_ner(data_dir: str | Path) -> Path:
    return ensure_pap_ner_data(data_dir, allow_download=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/pap_ner")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    ensure_pap_ner_data(args.data_dir, allow_download=not args.no_download)


if __name__ == "__main__":
    main()
