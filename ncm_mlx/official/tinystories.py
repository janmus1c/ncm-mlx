"""TinyStories data for training the released TinyStories architecture family.

Source: ``roneneldan/TinyStories`` on Hugging Face (CDLA-Sharing-1.0), pinned to
one revision and SHA-256 checked. We use the GPT-4-only V2 text files: the
upstream tokenized training stream (``hazyresearch/ncm-datasets``) starts with
the first V2 story, so this is the variant the released checkpoint appears to
have been trained on (an inference, not a documented fact).

Data contract (one row per story, the upstream generation contract):

* A story is the text between two ``<|endoftext|>`` markers, kept verbatim and
  prefixed with ``"\\n"`` when it does not already start with one (stories in
  the stream follow ``<|endoftext|>\\n``). Whitespace-only segments are dropped.
* Its tokens are ``bpe.encode(story) + [EOS]``, truncated to the first
  ``seq_len`` tokens and right-padded with PAD. Released samples start at a story
  opening and usually run to the 128-token limit without an EOS, which matches
  story-anchored windows; later windows of long stories are not used.
* The null prefix is a single EOS token (the previous story's end).
* Continuation task (``prefix_len`` P > 0): a row is the first ``P + seq_len`` tokens of the
  same story, the first P being the prefix and the next ``seq_len`` the target. Stories whose
  first P tokens are not all real tokens (story shorter than P + 1 tokens) are dropped from all
  splits. Split membership is unchanged: it is decided per story before the filter.
* Train rows come from the V2 train file. The V2 validation file is split by a
  hash of the story text into validation and test halves, and any story whose
  stripped text also occurs in the train file is removed from both, so held-out
  documents never overlap training documents.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
LICENSE = "CDLA-Sharing-1.0"
FILES = {
    "TinyStoriesV2-GPT4-train.txt": "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443",
    "TinyStoriesV2-GPT4-valid.txt": "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d",
}
TRAIN_FILE, VALID_FILE = FILES
SEPARATOR = "<|endoftext|>"
EOS_ID, PAD_ID = 0, 1
CHAR_WINDOW = 1600
CACHE_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path, log=print) -> None:
    """Download the pinned V2 text files into ``dest`` and verify their SHA-256."""
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = dest / name
        if target.exists() and sha256(target) == digest:
            log(f"ok      {target}")
            continue
        h = hashlib.sha256()
        with urlopen(f"{base}/{name}", timeout=300) as r, \
                tempfile.NamedTemporaryFile(dir=dest, delete=False) as f:
            tmp = Path(f.name)
            for chunk in iter(lambda: r.read(1 << 22), b""):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != digest:
            tmp.unlink()
            raise SystemExit(f"SHA-256 mismatch while downloading {name}")
        tmp.replace(target)
        log(f"fetched {target}")


def verify(raw: Path) -> None:
    for name, digest in FILES.items():
        path = raw / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m ncm_mlx official prepare-data` with network access")
        if sha256(path) != digest:
            raise SystemExit(f"SHA-256 mismatch for {path}")


def split_stories(text: str) -> list[str]:
    """Stories of one V2 file, in file order, in the form fed to the tokenizer."""
    out = []
    for seg in text.split(SEPARATOR):
        if not seg.strip():
            continue
        out.append(seg if seg.startswith("\n") else "\n" + seg)
    return out


def story_key(story: str) -> str:
    return hashlib.sha1(story.strip().encode("utf-8")).hexdigest()


def encode_window(bpe, story: str, seq_len: int, cache: dict | None = None) -> list[int]:
    """First ``seq_len`` ids of ``encode(story) + [EOS]``, without tokenizing the whole story.

    Pre-tokenization is local, so chunks well before a character cut are
    unchanged by it. The last two chunks before the cut are discarded and the
    full story is used when the remainder is too short.
    """
    from ncm_mlx.official.bpe import pre_tokenize

    def ids_of(chunks):
        out = []
        for w in chunks:
            hit = cache.get(w) if cache is not None else None
            if hit is None:
                hit = [bpe.ids[p] for p in bpe._merge(w.encode("utf-8"))]
                if cache is not None:
                    cache[w] = hit
            out += hit
        return out

    if len(story) > CHAR_WINDOW:
        ids = ids_of(pre_tokenize(story[:CHAR_WINDOW])[:-2])
        if len(ids) >= seq_len:
            return ids[:seq_len]
    ids = ids_of(pre_tokenize(story)) + [EOS_ID]
    return ids[:seq_len]


def _encode_shard(args) -> np.ndarray:
    from ncm_mlx.official.bpe import BPETokenizer

    bpe_path, stories, seq_len = args
    bpe = BPETokenizer.load(bpe_path)
    cache: dict = {}
    out = np.full((len(stories), seq_len), PAD_ID, dtype=np.uint16)
    for i, s in enumerate(stories):
        ids = encode_window(bpe, s, seq_len, cache)
        out[i, : len(ids)] = ids
    return out


def encode_all(bpe_path: Path, stories: list[str], seq_len: int, workers: int) -> np.ndarray:
    if not stories:
        return np.zeros((0, seq_len), dtype=np.uint16)
    size = max(1, min(20_000, len(stories) // max(workers, 1) + 1))
    shards = [(str(bpe_path), stories[i : i + size], seq_len) for i in range(0, len(stories), size)]
    if workers <= 1:
        return np.concatenate([_encode_shard(s) for s in shards])
    with ProcessPoolExecutor(workers) as ex:
        return np.concatenate(list(ex.map(_encode_shard, shards)))


def _atomic_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def split_dir(root: Path, seq_len: int, limit: int = 0, prefix_len: int = 0) -> Path:
    name = f"v2-bpe8192-L{seq_len}" + (f"-P{prefix_len}" if prefix_len else "")
    return root / (f"{name}-first{limit}" if limit else name)


def prepare(raw: Path, root: Path, bpe_path: Path, seq_len: int = 128, workers: int = 0,
            limit: int = 0, log=print, prefix_len: int = 0) -> dict:
    """Tokenize the pinned files into ``(N, prefix_len + seq_len)`` uint16 arrays; return the manifest.

    ``limit`` keeps only the first stories of each file (tests and smoke runs).
    """
    out = split_dir(root, seq_len, limit, prefix_len)
    manifest_path = out / "manifest.json"
    bpe_digest = sha256(bpe_path)
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if (m.get("cache_version") == CACHE_VERSION and m.get("bpe_sha256") == bpe_digest
                and m.get("prefix_len", 0) == prefix_len):
            return m
    if not limit:
        verify(raw)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    out.mkdir(parents=True, exist_ok=True)
    train = split_stories((raw / TRAIN_FILE).read_text(encoding="utf-8"))
    valid = split_stories((raw / VALID_FILE).read_text(encoding="utf-8"))
    if limit:
        train, valid = train[:limit], valid[: max(limit // 10, 20)]
    train_keys = {story_key(s) for s in train}
    held = [s for s in valid if story_key(s) not in train_keys]
    overlap = len(valid) - len(held)
    val = [s for s in held if int(story_key(s), 16) % 2 == 0]
    test = [s for s in held if int(story_key(s), 16) % 2 == 1]
    log(f"stories: train {len(train)}, validation {len(val)}, test {len(test)}, "
        f"removed {overlap} held-out stories also in train")
    counts = {}
    for name, stories in (("train", train), ("val", val), ("test", test)):
        arr = encode_all(bpe_path, stories, prefix_len + seq_len, workers)
        dropped = 0
        if prefix_len:
            keep = (arr[:, :prefix_len] > max(EOS_ID, PAD_ID)).all(1)
            dropped = int((~keep).sum())
            arr = arr[keep]
        _atomic_npy(out / f"{name}.npy", arr)
        real = (arr[:, prefix_len:] != PAD_ID).sum(1)
        counts[name] = {
            "stories": len(stories) - dropped,
            **({"dropped_shorter_than_prefix": dropped} if prefix_len else {}),
            "positions": int(arr[:, prefix_len:].size),
            "non_pad_tokens": int(real.sum()),
            "pad_fraction": round(float((arr[:, prefix_len:] == PAD_ID).mean()), 4),
            "with_eos_in_window": round(float((arr[:, prefix_len:] == EOS_ID).any(1).mean()), 4),
        }
        log(f"{name}: {counts[name]}")
    manifest = {
        "cache_version": CACHE_VERSION,
        "dataset": REPO, "revision": REVISION, "license": LICENSE,
        "files": {k: v for k, v in FILES.items()},
        "bpe_sha256": bpe_digest, "seq_len": seq_len, "eos_id": EOS_ID, "pad_id": PAD_ID,
        "limit": limit, "heldout_overlap_removed": overlap,
        **({"prefix_len": prefix_len} if prefix_len else {}),
        "split": "train file -> train; valid file -> val/test by sha1(story.strip()) parity",
        "splits": counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(directory: Path, name: str, mmap: bool = True) -> np.ndarray:
    return np.load(directory / f"{name}.npy", mmap_mode="r" if mmap else None)


def iter_texts(path: Path) -> Iterator[str]:
    yield from split_stories(path.read_text(encoding="utf-8"))
