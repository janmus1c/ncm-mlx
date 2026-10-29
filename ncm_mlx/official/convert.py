"""Convert released NCM checkpoints to MLX safetensors.

Every upstream tensor name is matched by exactly one explicit rule: it becomes
an MLX parameter (with a stated layout change), is checked against a value the
MLX model recomputes and then dropped, or is training-only state that inference
does not read. Any unmatched name, missing parameter or shape mismatch stops
the conversion. Downloads are checked against pinned SHA-256 digests before
they are parsed; ``.pt`` files are read by :mod:`torch_reader`, which refuses
all non-tensor globals, and the BPE pickles by an unpickler that refuses every
global.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from ncm_mlx.official import spec as specs
from ncm_mlx.official import torch_reader

Rule = tuple[str, str | None, Callable[[np.ndarray], np.ndarray] | None]


def _same(a: np.ndarray) -> np.ndarray:
    return a


def _conv_k3(w: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(w.transpose(0, 2, 1))


def _strided_down(w: np.ndarray) -> np.ndarray:
    out_c, in_c, s = w.shape
    return np.ascontiguousarray(w.transpose(2, 1, 0).reshape(s * in_c, out_c))


def _strided_up(w: np.ndarray) -> np.ndarray:
    in_c, out_c, s = w.shape
    return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(in_c, s * out_c))


_BLOCK = [
    (r"ln_1\.(weight|bias)", r"ln1.\1"),
    (r"attn\.c_attn\.(weight|bias)", r"attn.qkv.\1"),
    (r"attn\.c_proj\.(weight|bias)", r"attn.proj.\1"),
    (r"attn\.(q_norm|k_norm)\.weight", r"attn.\1.weight"),
    (r"ln_2\.(weight|bias)", r"ln2.\1"),
    (r"mlp\.c_fc\.(weight|bias)", r"fc.\1"),
    (r"mlp\.c_proj\.(weight|bias)", r"out.\1"),
]

CODEC_RULES: list[Rule] = [
    (r"encoder\.wte\.weight", "tok.weight", _same),
    (r"encoder\.wpe\.weight", "pos.weight", _same),
    (r"vq\.embeddings\.(\d+)\.weight", r"codebooks.\1", _same),
    (r"vq\.quant_resi\.(\d+)\.weight", r"phi.\1.weight", _conv_k3),
    (r"vq\.quant_resi\.(\d+)\.bias", r"phi.\1.bias", _same),
    (r"vq\.learned_down_l1\.(\d+)\.weight", r"down0.\1.weight", _strided_down),
    (r"vq\.learned_down_l1\.(\d+)\.bias", r"down0.\1.bias", _same),
    (r"vq\.learned_up_l1\.(\d+)\.weight", r"up0.\1.weight", _strided_up),
    (r"vq\.learned_up_l1\.(\d+)\.bias", r"up0.\1.bias", _same),
    *[(r"decoder\.middle_block\." + a, "dec_block." + b, _same) for a, b in _BLOCK],
    (r"decoder\.ln_f\.weight", "dec_ln.weight", _same),
    (r"decoder\.output_layer\.weight", "dec_head.weight", _same),
    (r"vq\.(ema_cluster_size|ema_w|codebook_hit)_\d+", None, None),
]

GENERATOR_RULES: list[Rule] = [
    (r"transformer\.wte\.(weight|bias)", r"in_proj.\1", _same),
    (r"transformer\.wle\.weight", "level_emb.weight", _same),
    *[(r"transformer\.blocks\.(\d+)\." + a, r"blocks.\1." + b.replace("\\1", "\\2"), _same)
      for a, b in _BLOCK],
    (r"transformer\.ln_f\.(weight|bias)", r"ln_f.\1", _same),
    (r"lm_heads\.(\d+)\.proj\.weight", r"heads.\1.proj.weight", _same),
    (r"lm_heads\.(\d+)\.log_inv_T", r"heads.\1.log_inv_t", _same),
    (r"lm_heads\.(\d+)\.(weight|bias)", r"heads.\1.\2", _same),
    (r"l0_queries", "l0_queries", _same),
    (r"lvl_1L|attn_mask_for_masking", None, None),
]


def map_state(state: dict[str, np.ndarray], rules: list[Rule]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Apply ``rules``; return MLX tensors and the upstream names that were not converted."""
    out, dropped = {}, []
    for name, value in state.items():
        hits = [(p, t, f) for p, t, f in rules if re.fullmatch(p, name)]
        if len(hits) != 1:
            raise ValueError(f"upstream tensor {name!r} matched {len(hits)} conversion rules")
        pattern, target, fn = hits[0]
        if target is None:
            dropped.append(name)
            continue
        key = re.sub(pattern, target, name)
        if key in out:
            raise ValueError(f"two upstream tensors map to {key!r}")
        out[key] = fn(np.asarray(value))
    return out, dropped


def expected_level_ids(levels: list[int]) -> np.ndarray:
    return np.concatenate([np.full(n, i) for i, n in enumerate(levels)])[None]


def expected_training_mask(prefix_len: int, levels: list[int]) -> np.ndarray:
    lvl = expected_level_ids(levels)[0]
    T, P = len(lvl), prefix_len
    allowed = np.zeros((P + T, P + T), dtype=bool)
    allowed[:, :P] = True
    allowed[P:, P:] = lvl[:, None] == lvl[None, :]
    return np.where(allowed, 0.0, -np.inf).astype(np.float32)[None, None]


def check_derived(state: dict, task: specs.TaskSpec) -> None:
    """The stored level ids and training mask must equal the ones the MLX generator assumes."""
    g = task.generator
    levels = list(task.codec.levels[0 if g.query_level0 else 1:])
    if not np.array_equal(state["lvl_1L"], expected_level_ids(levels)):
        raise ValueError("checkpoint level ids differ from the configured levels")
    if not np.array_equal(state["attn_mask_for_masking"], expected_training_mask(g.prefix_len, levels)):
        raise ValueError("checkpoint attention mask is not the block-diagonal mask for this config")


def _check_shapes(model, weights: dict[str, np.ndarray], what: str) -> None:
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.parameters()))
    missing = sorted(set(params) - set(weights))
    extra = sorted(set(weights) - set(params))
    if missing or extra:
        raise ValueError(f"{what}: missing {missing[:8]} unexpected {extra[:8]}")
    for k, v in weights.items():
        if tuple(params[k].shape) != v.shape:
            raise ValueError(f"{what}: {k} has shape {v.shape}, model expects {tuple(params[k].shape)}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(task: specs.TaskSpec, src: Path) -> None:
    for name, digest in task.files.items():
        path = src / name
        if not path.exists():
            raise SystemExit(f"missing {path}; run `python -m ncm_mlx official fetch --task {task.name}`")
        if sha256(path) != digest:
            raise SystemExit(f"SHA-256 mismatch for {path}; expected the pinned release file")


def fetch(task: specs.TaskSpec, dest: Path) -> None:
    """Download the pinned release files for ``task`` into ``dest`` and verify them."""
    base = f"https://huggingface.co/{specs.REPO}/resolve/{specs.REVISION}/{task.remote_dir}"
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in task.files.items():
        target = dest / name
        if target.exists() and sha256(target) == digest:
            print(f"ok      {target}")
            continue
        h = hashlib.sha256()
        with urlopen(f"{base}/{name}", timeout=120) as r, \
                tempfile.NamedTemporaryFile(dir=dest, delete=False) as f:
            tmp = Path(f.name)
            for chunk in iter(lambda: r.read(1 << 20), b""):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != digest:
            tmp.unlink()
            raise SystemExit(f"SHA-256 mismatch while downloading {name}")
        tmp.replace(target)
        print(f"fetched {target}")


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def _read_plain_pickle(path: Path):
    with open(path, "rb") as f:
        return _NoGlobals(f).load()


def convert_bpe(src: Path, out: Path) -> None:
    from ncm_mlx.official.bpe import BPETokenizer

    vocab = _read_plain_pickle(src / "bpe_vocab_8192.pickle")
    merges = _read_plain_pickle(src / "bpe_merges_8192.pickle")
    if not (isinstance(vocab, dict) and sorted(vocab) == list(range(len(vocab)))
            and all(isinstance(v, bytes) for v in vocab.values())):
        raise ValueError("BPE vocabulary is not a contiguous id -> bytes mapping")
    if not all(isinstance(m, tuple) and len(m) == 2 and all(isinstance(b, bytes) for b in m) for m in merges):
        raise ValueError("BPE merges are not byte pairs")
    BPETokenizer([vocab[i] for i in range(len(vocab))], list(merges),
                 ["<|endoftext|>", "<|pad|>"]).save(out / "bpe.json")


def convert(task: specs.TaskSpec, src: Path, out: Path) -> dict:
    """Convert ``src`` (release files) into ``out`` (MLX files); return a summary."""
    import mlx.core as mx

    from ncm_mlx.official.model import Codec, Generator

    verify_files(task, src)
    out.mkdir(parents=True, exist_ok=True)
    tok_ck = torch_reader.load(src / "tokenizer.pt")
    gen_ck = torch_reader.load(src / "generator.pt")
    if set(tok_ck) != {"model"} or set(gen_ck) - {"model", "inference"}:
        raise ValueError(f"unexpected checkpoint sections {sorted(tok_ck)} / {sorted(gen_ck)}")
    codec_w, codec_drop = map_state(tok_ck["model"], CODEC_RULES)
    check_derived(gen_ck["model"], task)
    gen_w, gen_drop = map_state(gen_ck["model"], GENERATOR_RULES)

    codec = Codec(task.codec)
    _check_shapes(codec, codec_w, "codec")
    gen = Generator(task.generator, codec)
    _check_shapes(gen, gen_w, "generator")

    mx.save_safetensors(str(out / "codec.safetensors"), {k: mx.array(v) for k, v in codec_w.items()})
    mx.save_safetensors(str(out / "generator.safetensors"), {k: mx.array(v) for k, v in gen_w.items()})
    extras = {}
    inference = gen_ck.get("inference", {})
    if set(inference) - {"level0_unigram"}:
        raise ValueError(f"unexpected inference entries {sorted(inference)}")
    if "level0_unigram" in inference:
        p = np.asarray(inference["level0_unigram"], dtype=np.float32)
        if p.shape != (task.codec.codebook_sizes[0],) or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-4):
            raise ValueError("level-0 unigram is not a distribution over level-0 codes")
        extras["level0_unigram"] = mx.array(p)
    if not task.generator.query_level0 and "level0_unigram" not in extras:
        raise ValueError("unconditional level 0 needs the stored level-0 unigram")
    if extras:
        mx.save_safetensors(str(out / "inference.safetensors"), extras)
    if task.tokenizer == "bpe":
        convert_bpe(src, out)
    meta = {"task": task.to_json(), "source": {"repo": specs.REPO, "revision": specs.REVISION,
                                               "files": task.files},
            "dropped_upstream_tensors": sorted(codec_drop + gen_drop)}
    (out / "spec.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"codec_tensors": len(codec_w), "generator_tensors": len(gen_w),
            "dropped": len(codec_drop) + len(gen_drop), "out": str(out)}
