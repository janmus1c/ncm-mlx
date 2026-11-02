"""Compare the MLX codec with upstream PyTorch on the codec-only XSum and WMT14 checkpoints.

For each input row of token ids both implementations run on CPU in fp32:

1. Upstream ``VQVAE`` encodes to per-level codes; the MLX codec encodes the same ids.
2. The decoder is fed each level's cumulative reconstruction built from upstream's codes, so a
   code disagreement at one level cannot hide a decoder disagreement at another.

Upstream is also run row by row and as one batch. Differing results there would show that the
reference itself is not batch-invariant, which matters when reading a single near-tie code.

Needs PyTorch and upstream's dependencies only for this run, e.g.::

    uv run --with torch --with omegaconf --with hydra-core --with regex \\
        python scripts/official_codec_parity.py --upstream /path/to/HazyResearch-ncm --task xsum

Inputs are the ``token_ids`` of ``--inputs`` (default: the committed fixture). ``--fixture`` writes
inputs plus upstream's codes and decoded ids to the fixture used by ``tests/test_official_codec.py``.

``--synthetic`` needs no checkpoint: it loads seeded random tensors of a narrow codec with the real
level geometry (``tests/codec_state.py``) into upstream's ``VQVAE`` with ``strict=True`` and stores
upstream's codes for them, which the CPU test reproduces with this port's converter and codec.
Upstream code is imported from ``--upstream`` and is not part of this repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIXTURE = ROOT / "tests/fixtures/official_codec_parity.json"
CONFIGS = ROOT / "tests/fixtures/official_codec_configs.json"
SYNTHETIC_SEED = 20261102


def upstream_codec(upstream: Path, config: dict, state: dict):
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(upstream))
    from models.vqvae import VQVAE

    torch.manual_seed(0)
    vqvae = VQVAE(**OmegaConf.create(config))
    vqvae.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True)
    return vqvae.eval()


def upstream_reference(vqvae, rows: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Flat ``(B, sum(levels))`` codes and, per level, the decoder's argmax ids from the cumulative latent."""
    import torch

    with torch.no_grad():
        z = vqvae.encode_z(torch.tensor(rows, dtype=torch.long))
        flat, zhats = vqvae.vq.z_to_idx(z, return_zhats=True)
        return flat.numpy(), [vqvae.decoder(h).argmax(-1).numpy() for h in zhats]


def synthetic(args, task) -> dict:
    """Upstream codes for seeded narrow-codec tensors; the committed reference for ``tests/test_official_codec.py``."""
    from codec_state import small, upstream_state

    spec_ = small(task.codec)
    config = {**json.loads(CONFIGS.read_text())[task.name]["vqvae"], "embed_dim": spec_.dim,
              "in_vocab_size": spec_.vocab_size, "vqvae_vocab_size": list(spec_.codebook_sizes)}
    vqvae = upstream_codec(Path(args.upstream), config, upstream_state(spec_, SYNTHETIC_SEED))
    rows = np.random.default_rng(SYNTHETIC_SEED + 1).integers(0, spec_.vocab_size, (4, spec_.seq_len))
    flat, decoded = upstream_reference(vqvae, rows)
    fixture = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
    fixture.setdefault(task.name, {})["synthetic"] = {
        "seed": SYNTHETIC_SEED, "token_ids": rows.tolist(), "codes": flat.tolist(),
        "decoded": [d.tolist() for d in decoded]}
    FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    return {"task": task.name, "synthetic_rows": len(rows), "codes": int(flat.size)}


def run(args) -> dict:
    import mlx.core as mx
    import torch

    from ncm_mlx.official import sample, spec, torch_reader

    mx.set_default_device(mx.cpu)
    task = spec.get(args.task)
    if task.generator is not None:
        raise SystemExit(f"{task.name} has a generator; use scripts/official_parity.py")
    if args.synthetic:
        sys.path.insert(0, str(ROOT / "tests"))
        return synthetic(args, task)
    src = Path(args.src or ROOT / "artifacts/official" / task.remote_dir)
    fixture = json.loads(FIXTURE.read_text()) if FIXTURE.exists() else {}
    inputs = json.loads(Path(args.inputs).read_text()) if args.inputs else fixture[task.name]["inputs"]
    rows = np.array(inputs["token_ids"], dtype=np.int64)
    levels = list(task.codec.levels)
    offsets = np.cumsum([0, *levels])

    config = json.loads(CONFIGS.read_text())[task.name]["vqvae"]
    state = torch_reader.load(src / "tokenizer.pt")["model"]
    vqvae = upstream_codec(Path(args.upstream), config, state)
    flat, decoded = upstream_reference(vqvae, rows)
    single = [upstream_reference(vqvae, r[None]) for r in rows]
    flat_single = np.concatenate([f for f, _ in single])
    _, codec = sample.load_codec(src / "mlx")

    codes = [np.array(c) for c in codec.encode(mx.array(rows.astype(np.int32)))]
    report = {"task": task.name, "torch": torch.__version__, "mlx": mx.__version__, "device": "cpu",
              "dtype": "float32", "upstream_code_revision": spec.UPSTREAM_CODE_REVISION,
              "checkpoint_revision": spec.REVISION, "inputs": len(rows), "levels": len(levels),
              "upstream_batch_vs_single_code_differences": int((flat != flat_single).sum()),
              "code_mismatches": [], "forced_decoded_mismatches": 0}
    upstream_levels = [flat[:, offsets[k]:offsets[k + 1]] for k in range(len(levels))]
    rest = codec.embed(mx.array(rows.astype(np.int32)))
    for k in range(len(levels)):
        # Distances of the greedy search under upstream's own residual, to size any disagreement.
        x = codec._down(rest, k) if k != len(levels) - 1 else rest
        cb = codec.codebooks[k]
        dist = np.array((x * x).sum(-1, keepdims=True) + (cb * cb).sum(-1) - 2 * (x @ cb.T))
        for b, p in np.argwhere(codes[k] != upstream_levels[k]):
            mine, theirs = int(codes[k][b, p]), int(upstream_levels[k][b, p])
            report["code_mismatches"].append({
                "input": int(b), "level": k, "position": int(p), "mlx": mine, "upstream": theirs,
                "distance_gap": float(abs(dist[b, p, mine] - dist[b, p, theirs])),
                "distance_scale": float(np.abs(dist[b, p]).max())})
        forced = mx.array(upstream_levels[k].astype(np.int32))
        rest = rest - codec.level_contribution(forced, k)
    for k in range(len(levels)):
        cumulative = codec.cumulative([mx.array(u.astype(np.int32)) for u in upstream_levels[:k + 1]])[-1]
        ids = np.array(mx.argmax(codec.decode_logits(cumulative), axis=-1))
        report["forced_decoded_mismatches"] += int((ids != decoded[k]).sum())
    if args.fixture:
        fixture.setdefault(task.name, {}).update({
            "inputs": {"labels": inputs["labels"], "token_ids": inputs["token_ids"]},
            "codes": flat.tolist(), "decoded": [d.tolist() for d in decoded]})
        fixture["_meta"] = {**{k: report[k] for k in ("torch", "upstream_code_revision", "checkpoint_revision")},
                            "source_tokenizers": spec.SOURCE_TOKENIZERS}
        FIXTURE.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="checkout of HazyResearch/ncm")
    ap.add_argument("--task", choices=["xsum", "wmt14"], required=True)
    ap.add_argument("--src", help="release files (default artifacts/official/<task dir>); converted weights in mlx/")
    ap.add_argument("--inputs", help="JSON with labels and token_ids (default: the committed fixture)")
    ap.add_argument("--synthetic", action="store_true", help="narrow random tensors, no checkpoint; writes the fixture")
    ap.add_argument("--fixture", action="store_true", help="write inputs and upstream references to the fixture")
    ap.add_argument("--out", help="write the report JSON here")
    args = ap.parse_args()
    report = run(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
