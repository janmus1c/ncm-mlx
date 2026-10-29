"""Load converted official checkpoints and generate level by level."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ncm_mlx.official import spec as specs
from ncm_mlx.official.bpe import BPETokenizer
from ncm_mlx.official.model import Codec, Generator

SUDOKU_CHARS = "_123456789|B"


@dataclass
class Official:
    task: specs.TaskSpec
    codec: Codec
    gen: Generator
    bpe: BPETokenizer | None
    level0_unigram: mx.array | None

    def decode(self, tokens) -> str:
        """Text for one row of decoder tokens, following the upstream evaluation's cleanup."""
        ids = [int(t) for t in tokens]
        if self.task.tokenizer == "sudoku":
            return "".join(SUDOKU_CHARS[t] for t in ids if t != 11)
        eos, pad = self.task.eos_id, self.task.pad_id
        if eos in ids:
            ids = ids[: ids.index(eos)]
        return self.bpe.decode([t for t in ids if t not in (eos, pad)], skip_special=True)


def load(directory: str | Path) -> Official:
    d = Path(directory)
    if not (d / "spec.json").exists():
        raise SystemExit(f"no converted checkpoint in {d}; run `python -m ncm_mlx official convert` first")
    task = specs.from_json(json.loads((d / "spec.json").read_text())["task"])
    codec = Codec(task.codec)
    codec.load_weights(str(d / "codec.safetensors"), strict=True)
    gen = Generator(task.generator, codec)
    gen.load_weights(str(d / "generator.safetensors"), strict=True)
    codec.eval()
    gen.eval()
    unigram = None
    if (d / "inference.safetensors").exists():
        unigram = mx.load(str(d / "inference.safetensors"))["level0_unigram"]
    bpe = BPETokenizer.load(d / "bpe.json") if task.tokenizer == "bpe" else None
    return Official(task, codec, gen, bpe, unigram)


def sample_codes(logits: mx.array, temperature: float, top_k: int, top_p: float, key) -> mx.array:
    """Upstream sampling semantics: greedy at temperature 0, else temperature, top-k, nucleus."""
    if temperature == 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k > 0:
        kth = mx.sort(logits, axis=-1)[..., -top_k][..., None]
        logits = mx.where(logits < kth, -mx.inf, logits)
    if top_p < 1.0:
        order = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, order, axis=-1)
        probs = mx.softmax(sorted_logits, axis=-1)
        remove = (mx.cumsum(probs, axis=-1) - probs) > top_p
        sorted_logits = mx.where(remove, -mx.inf, sorted_logits)
        logits = mx.put_along_axis(mx.zeros_like(logits), order, sorted_logits, axis=-1)
    return mx.random.categorical(logits, key=key)


@dataclass
class Generation:
    codes: list[mx.array]
    tokens: mx.array
    per_level_tokens: list[mx.array] | None
    level_logits: list[mx.array] | None


def generate(m: Official, prefix_tokens: mx.array, temperature: float = 1.0, top_k: int = 0,
             top_p: float = 1.0, seed: int = 0, level0: mx.array | None = None,
             keep_levels: bool = False, keep_logits: bool = False,
             forced: list[mx.array] | None = None) -> Generation:
    """Sample every level with one generator forward each; decode the final reconstruction.

    Level 0 comes from ``level0`` when given, else from learned level-0 queries
    (Sudoku) or the stored level-0 unigram (TinyStories). ``forced`` replaces
    each level's sampled codes by given codes (teacher forcing through the same
    per-level forwards), which parity checks use to compare logits.
    """
    codec, gen = m.codec, m.gen
    keys = mx.random.split(mx.random.key(seed), len(codec.levels) + 1)
    B = prefix_tokens.shape[0]
    prefix_emb = gen.prefix_embedding(prefix_tokens)
    logits_seen = []

    def step(j, level_input, key):
        logits = gen.level_logits(j, level_input, prefix_emb, prefix_tokens)
        if keep_logits:
            logits_seen.append(logits)
        if forced is not None:
            return forced[j + gen.first]
        return sample_codes(logits, temperature, top_k, top_p, key)

    if level0 is None and forced is not None and not gen.spec.query_level0:
        level0 = forced[0]
    if level0 is not None:
        c0 = level0
    elif gen.spec.query_level0:
        c0 = step(0, mx.broadcast_to(gen.l0_queries, (B, *gen.l0_queries.shape[1:])), keys[0])
    else:
        if m.level0_unigram is None:
            raise ValueError("this checkpoint needs level-0 codes or a level-0 unigram")
        c0 = mx.random.categorical(mx.log(m.level0_unigram), shape=(B, codec.levels[0]), key=keys[0])
    codes = [c0]
    z = codec.level_contribution(c0, 0)
    per_level = [z] if keep_levels else None
    for k in range(1, len(codec.levels)):
        c = step(k - gen.first, codec.pool(z, k), keys[k])
        codes.append(c)
        z = z + codec.level_contribution(c, k)
        if keep_levels:
            per_level.append(z)
    tokens = mx.argmax(codec.decode_logits(z), axis=-1)
    level_tokens = [mx.argmax(codec.decode_logits(zk), axis=-1) for zk in per_level] if keep_levels else None
    return Generation(codes, tokens, level_tokens, logits_seen if keep_logits else None)


def unconditional_prefix(m: Official, batch: int) -> mx.array:
    """TinyStories samples start from a single end-of-text prefix token, as in the upstream evaluation."""
    return mx.full((batch, m.task.generator.prefix_len), m.task.eos_id, dtype=mx.int32)
