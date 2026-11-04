"""Record and render level-by-level demos from trained checkpoints: Sudoku, TinyStories and NCM against AR.

Two steps, both in one command. ``sudoku`` / ``story`` run the checkpoint once and write a provenance
JSON next to the figures (checkpoint hashes, seed, board or sample, the decoded output after every level);
the SVGs are then drawn from that JSON alone, so ``--render-only`` redraws them without a model. Only real
model outputs are drawn; nothing is edited by hand.

Swap in a better model by rerunning one command with a new ``--name`` (GitHub caches images for five
minutes, so a changed figure needs a new file name)::

    NCM_MLX_MEMORY_LIMIT_GB=8 python scripts/render_pass_demos.py sudoku \\
        --config configs/sudoku-e19-funded.json --name sudoku-12-passes
    NCM_MLX_MEMORY_LIMIT_GB=8 python scripts/render_pass_demos.py story \\
        --weights path/to/tinystories-run/export --label "independently trained 87M MLX NCM" \\
        --eval-note "GPT-2 large generative perplexity 24.40" --name story-12-passes-87m
    NCM_MLX_MEMORY_LIMIT_GB=8 python scripts/render_pass_demos.py split \\
        --ncm path/to/tinystories-run/export --ar path/to/tinystories-run/export/ar.safetensors \\
        --name level-vs-token-87m

``split`` samples the NCM unconditionally, keeps the first sample whose first sentence is well formed, replays its
tokens through the AR baseline one call per token (forced decoding, recording the AR's log-probability of every
token) and draws the same text in both panels of a split screen driven by a shared model-call counter.

Outputs in ``docs/assets``: ``NAME.json``, ``NAME-{light,dark}.svg`` and ``NAME-mobile-{light,dark}.svg``
(static strips) and the same four with ``-animated`` after NAME (levels appear one after another; static
under reduced motion).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from html import escape
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_readme_hero import FONT, THEMES, animation_css, excerpt, level_dots, words_markup, wrap

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime() -> dict:
    import mlx.core as mx

    from ncm_mlx.device import runtime_info

    info = runtime_info()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return {"mlx": mx.__version__, "device": info["device"], "chip": info.get("chip"),
            "ncm_mlx_commit": commit.stdout.strip() or None, "command": " ".join(sys.argv)}


# ------------------------------------------------------------------------------------------ recording


def record_sudoku(args) -> dict:
    import mlx.core as mx

    from ncm_mlx.config import Config
    from ncm_mlx.device import require_gpu
    from ncm_mlx.generation.sampler import SampleOptions, generate
    from ncm_mlx.text import sudoku, upstream_sudoku
    from ncm_mlx.text.tokenizer import CharTokenizer
    from ncm_mlx.training.common import build_codec, build_generator, load_model, param_count

    require_gpu()
    cfg = Config.load(args.config)
    out = Path(cfg.out_dir)
    gen_path = Path(args.weights) if args.weights else out / "generator.safetensors"
    codec_path = Path(args.codec) if args.codec else Path(cfg.codec_checkpoint or out / "codec.safetensors")
    tok = CharTokenizer("0123456789", "0", 81)
    codec = load_model(build_codec(cfg, tok.vocab_size), codec_path)
    gen = load_model(build_generator(cfg), gen_path)
    puzzles, solutions = upstream_sudoku.load(Path(args.boards), args.board_set)

    def solve(prefix: np.ndarray):
        return generate(gen, codec, prefix.shape[0], SampleOptions(temperature=0.0, seed=0),
                        keep_levels=True, prefix_tokens=mx.array(prefix))

    solved = np.zeros(len(puzzles), dtype=bool)
    for i in range(0, len(puzzles), 250):
        pred = np.array(solve(puzzles[i : i + 250]).tokens)
        solved[i : i + 250] = [sudoku.score(p, y, s)["solved"] for p, y, s in zip(puzzles[i : i + 250], pred, solutions[i : i + 250])]
    givens = (puzzles != 0).sum(axis=1)
    if args.board is not None:
        order = [args.board]
    else:
        order = [int(i) for i in np.lexsort((np.arange(len(puzzles)), givens)) if solved[i]]
    for index in order:
        g = solve(puzzles[index : index + 1])
        final = np.array(g.tokens[0])
        if sudoku.score(puzzles[index], final, solutions[index])["solved"]:
            break
    else:
        raise SystemExit("no board solved in the single-board run")
    levels = [np.array(t[0]) for t in g.per_level_tokens]
    sizes = list(cfg.codec.levels)
    return {
        "kind": "sudoku",
        "title": "Sudoku in 12 passes",
        "model": f"independently trained {param_count(gen) / 1e6:.2f}M MLX NCM "
                 f"({cfg.name}), not the released HazyResearch checkpoint",
        "config": args.config,
        "checkpoint": {"generator": str(gen_path), "generator_sha256": sha256(gen_path),
                       "codec": str(codec_path), "codec_sha256": sha256(codec_path)},
        "decoding": {"temperature": 0.0, "seed": 0, "passes": len(levels), "level_sizes": sizes},
        "board_set": f"upstream {args.board_set} (hazyresearch/ncm-datasets)",
        "board_index": index,
        "board_rule": "explicit --board" if args.board is not None else
                      "fewest givens among boards the model solves, ties by lowest index",
        "set_solve_rate": round(float(solved.mean()), 4), "set_size": len(puzzles),
        "givens": int(givens[index]),
        "puzzle": "".join(map(str, puzzles[index])),
        "solution": "".join(map(str, solutions[index])),
        "passes": ["".join(map(str, lv)) for lv in levels],
        "final": "".join(map(str, final)),
        "runtime": runtime(),
    }


def record_story(args) -> dict:
    from ncm_mlx.device import require_gpu
    from ncm_mlx.official import sample

    require_gpu()
    weights = Path(args.weights)
    m = sample.load(weights)
    g = sample.generate(m, sample.unconditional_prefix(m, args.n), temperature=args.temperature,
                        seed=args.seed, keep_levels=True)
    sizes = list(m.codec.levels)
    finals = [m.decode(np.array(g.tokens[i])) for i in range(args.n)]
    index = args.sample
    per_level = [m.decode(np.array(lt[index])) for lt in g.per_level_tokens]
    if per_level[-1] != finals[index]:
        raise SystemExit("the last level must equal the final decode")
    return {
        "kind": "story",
        "title": "A story in 12 passes",
        "model": args.label,
        "evaluation": args.eval_note,
        "checkpoint": {"directory": str(weights),
                       **{f"{n}_sha256": sha256(weights / f"{n}.safetensors") for n in ("generator", "codec", "inference")}},
        "decoding": {"temperature": args.temperature, "seed": args.seed, "samples": args.n, "sample_index": index,
                     "passes": len(sizes), "level_sizes": sizes, "task": "unconditional TinyStories"},
        "sample_rule": args.sample_rule,
        "passes": per_level,
        "all_finals": finals,
        "runtime": runtime(),
    }


FIRST_SENTENCE = re.compile(r".*?[.!?](?=\s|$)", re.DOTALL)


def well_formed(text: str) -> bool:
    """Fixed sample rule: the first sentence starts with a capital, has 5 to 30 words, ends in . ! or ?,
    and has no repeated word, doubled punctuation or stray characters."""
    m = FIRST_SENTENCE.match(" ".join(text.split()))
    if not m:
        return False
    s = m.group(0)
    words = re.findall(r"[A-Za-z']+", s.lower())
    return (s[0].isupper() and 5 <= len(s.split()) <= 30 and re.fullmatch(r"[A-Za-z',\"!?. ]+", s) is not None
            and not re.search(r"[,.!?]{2}| [,.!?]", s) and all(a != b for a, b in pairwise(words)))


def set_memory_limit() -> None:
    import os

    import mlx.core as mx

    gb = float(os.environ.get("NCM_MLX_MEMORY_LIMIT_GB", "8"))
    mx.set_memory_limit(int(gb * 2**30))
    mx.set_cache_limit(int(min(gb / 8, 2) * 2**30))


def token_pieces(decode, ids: list[int]) -> list[str]:
    """Split a decoded token row into the text each token adds, so that the pieces join to the full decode."""
    pieces, prev = [], ""
    for k in range(1, len(ids) + 1):
        cur = decode(ids[:k])
        if cur.startswith(prev):
            pieces.append(cur[len(prev) :])
            prev = cur
        else:
            pieces.append("")
    full = decode(ids)
    if "".join(pieces) != full:
        pieces[-1] += full[len("".join(pieces)) :]
    return pieces


def forced_logprobs(ar, ids: list[int]) -> tuple[list[float], float]:
    """Feed ``ids`` one per call through the AR's key/value-cached step (forced decoding).

    Returns the log-probability the AR gives each token at the call that writes it, and the largest gap to a
    full causal forward over the same row as a check that the step is the model's own next-token prediction.
    """
    import mlx.core as mx

    cache = [None] * len(ar.blocks)
    tok = mx.full((1, 1), ar.bos_id, dtype=mx.int32)
    out = []
    for pos, t in enumerate(ids):
        logits = ar._cached(tok, pos, cache).astype(mx.float32)
        out.append(float((logits - mx.logsumexp(logits, axis=-1, keepdims=True))[0, t]))
        tok = mx.array([[t]], dtype=mx.int32)
    full = ar(mx.array([ids], dtype=mx.int32)).astype(mx.float32)
    full = full - mx.logsumexp(full, axis=-1, keepdims=True)
    ref = np.array(full[0])[np.arange(len(ids)), ids]
    return out, float(np.abs(ref - np.array(out)).max())


def record_split(args) -> dict:
    from ncm_mlx.device import require_gpu
    from ncm_mlx.official import sample
    from ncm_mlx.official.ar import TextAR
    from ncm_mlx.official.train_config import TextConfig

    require_gpu()
    set_memory_limit()
    ncm_dir, ar_path = Path(args.ncm), Path(args.ar)
    ar_meta_path = Path(args.ar_config) if args.ar_config else ar_path.with_suffix(".json")
    m = sample.load(ncm_dir)
    g = sample.generate(m, sample.unconditional_prefix(m, args.n), temperature=args.temperature,
                        seed=args.seed, keep_levels=True)
    ncm_finals = [m.decode(np.array(g.tokens[i])) for i in range(args.n)]
    ncm_index = next((i for i, t in enumerate(ncm_finals) if well_formed(t)), None)
    if ncm_index is None:
        raise SystemExit("no NCM sample of this seed has a well-formed first sentence")
    passes = [m.decode(np.array(lt[ncm_index])) for lt in g.per_level_tokens]
    if passes[-1] != ncm_finals[ncm_index]:
        raise SystemExit("the last level must equal the final decode")
    pass_tokens = [np.array(lt[ncm_index]).tolist() for lt in g.per_level_tokens]
    ids = np.array(g.tokens[ncm_index]).tolist()
    if ids != pass_tokens[-1]:
        raise SystemExit("the last level's tokens must equal the final tokens")
    eos, pad = m.task.eos_id, m.task.pad_id
    end = ids.index(eos) if eos in ids else len(ids)
    pad_in_text = sum(t == pad for t in ids[:end]) if pad is not None else 0

    cfg = TextConfig.from_dict(json.loads(ar_meta_path.read_text())["config"])
    gc = cfg.generator
    ar = TextAR(8192, cfg.data.seq_len, gc.dim, gc.n_layer, gc.n_head, gc.rope_base, prefix_len=cfg.data.prefix_len)
    ar.load_weights(str(ar_path), strict=True)
    ar.eval()
    if ar.prefix_len:
        raise SystemExit("the split figure needs an unconditional AR baseline (prefix_len 0)")
    if cfg.data.seq_len != len(ids):
        raise SystemExit(f"AR window {cfg.data.seq_len} differs from the NCM window {len(ids)}")
    logprobs, gap = forced_logprobs(ar, ids)
    nll = -float(np.mean(logprobs[:end])) if end else None
    first = m.gen.first
    return {
        "kind": "split",
        "title": "One level at a time vs one token at a time",
        "model": args.label,
        "text": {
            "tokens": ids,
            "window": len(ids),
            "length": end,
            "eos_at_token": None if end == len(ids) else end + 1,
            "pad_tokens_in_text": pad_in_text,
            "pieces": token_pieces(m.decode, ids),
            "decoded": ncm_finals[ncm_index],
            "cut_rule": ("decoded as the repository does (ncm_mlx.official.sample.Official.decode): cut at the first "
                         "end-of-story token, drop padding, BPE decode; whitespace runs are collapsed to one space "
                         "for display, identically in both panels"),
        },
        "ncm": {
            "checkpoint": {"directory": str(ncm_dir),
                           **{f"{n}_sha256": sha256(ncm_dir / f"{n}.safetensors")
                              for n in ("generator", "codec", "inference")}},
            "sample_index": ncm_index,
            "level_sizes": list(m.codec.levels),
            "calls_at_level": [max(k - first + 1, 0) for k in range(len(m.codec.levels))],
            "level0": "stored unigram, no generator call" if first else "learned queries, one generator call",
            "passes": passes,
            "pass_tokens": pass_tokens,
            "pass_pieces": [token_pieces(m.decode, t) for t in pass_tokens],
            "all_finals": ncm_finals,
        },
        "ar": {
            "checkpoint": {"weights": str(ar_path), "weights_sha256": sha256(ar_path),
                           "config": str(ar_meta_path), "parameters": json.loads(ar_meta_path.read_text()).get("parameters")},
            "mode": ("forced decoding: the NCM sample's tokens are fed one per call through the AR's key/value-cached "
                     "step (BOS first), so call n writes token n of the same text; not an independent AR sample"),
            "calls": len(ids),
            "token_logprobs": [round(x, 5) for x in logprobs],
            "text_logprob": round(float(np.sum(logprobs[:end])), 4),
            "text_mean_nll": None if nll is None else round(nll, 4),
            "text_perplexity": None if nll is None else round(float(np.exp(nll)), 3),
            "logprob_note": ("Natural-log probabilities the AR assigns to each NCM token at the call that writes it; "
                             "the summary covers the text up to the end-of-story token. They show how the AR scores "
                             "this text and are not drawn in the figure."),
            "step_vs_full_forward_max_abs_diff": round(gap, 6),
        },
        "decoding": {"temperature": args.temperature, "seed": args.seed, "samples": args.n,
                     "task": "unconditional TinyStories", "seq_len": cfg.data.seq_len},
        "sample_rule": ("The first of the seed's NCM samples (lowest index) whose first sentence is well formed: it "
                        "starts with a capital, has 5 to 30 words, ends in . ! or ? and has no repeated word, doubled "
                        "punctuation or stray characters. The AR replays that sample."),
        "counting": ("Model calls are forward passes of the transformer that writes the output: one per generated "
                     "level for NCM, one per token for AR. They are counted, not timed; an NCM call is larger "
                     "than an AR call, and NCM adds a codec decode at the end."),
        "runtime": runtime(),
    }


# ------------------------------------------------------------------------------------------ drawing


def head(w: int, h: int, title: str, desc: str, style: str, c: dict) -> list[str]:
    return [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title description">'
            f'<title id="title">{escape(title)}</title><desc id="description">{escape(desc)}</desc>'),
            style, f'<rect width="{w}" height="{h}" rx="16" fill="{c["canvas"]}"/>',
            f'<g font-family="{FONT}" fill="{c["ink"]}">']


def rule(x1: float, x2: float, y: float, c: dict) -> str:
    return (f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{c["rule"]}" stroke-width="2" '
            'stroke-linecap="round" stroke-dasharray="0 6"/>')


def caption_lines(out: list[str], lines: list[str], x: float, y: float, size: int, lead: int, c: dict) -> float:
    for j, line in enumerate(lines):
        out.append(f'<text x="{x}" y="{y + j * lead}" font-size="{size}" fill="{c["muted"]}">{escape(line)}</text>')
    return y + len(lines) * lead


def sudoku_tile(x: float, y: float, cell: float, board: str, puzzle: str, solution: str, c: dict) -> str:
    """Nine-by-nine grid: givens in solid ink, correct model cells in ink, wrong cells outlined, blanks as dots."""
    parts = [f'<rect x="{x}" y="{y}" width="{cell * 9}" height="{cell * 9}" rx="2" fill="{c["panel"]}"/>']
    for i in range(10):
        wide = i % 3 == 0
        parts.append(f'<line x1="{x + i * cell}" y1="{y}" x2="{x + i * cell}" y2="{y + 9 * cell}" '
                     f'stroke="{c["muted"] if wide else c["rule"]}" stroke-width="{1.4 if wide else 0.6}"/>')
        parts.append(f'<line x1="{x}" y1="{y + i * cell}" x2="{x + 9 * cell}" y2="{y + i * cell}" '
                     f'stroke="{c["muted"] if wide else c["rule"]}" stroke-width="{1.4 if wide else 0.6}"/>')
    fs = cell * 0.66
    for k, ch in enumerate(board):
        cx, cy = x + (k % 9) * cell + cell / 2, y + (k // 9) * cell + cell / 2
        given = puzzle[k] != "0"
        wrong = not given and ch != solution[k]
        if wrong:
            parts.append(f'<rect x="{cx - cell / 2 + 1}" y="{cy - cell / 2 + 1}" width="{cell - 2}" height="{cell - 2}" '
                         f'fill="none" stroke="{c["ink"]}" stroke-width="1.2"/>')
        if ch == "0":
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.07}" fill="{c["muted"]}"/>')
        else:
            parts.append(f'<text x="{cx}" y="{cy + fs * 0.35}" font-size="{fs:.1f}" text-anchor="middle" '
                         f'font-weight="{700 if given else 400}" fill="{c["ink"] if given or not wrong else c["muted"]}">{ch}</text>')
    return "".join(parts)


def sudoku_svg(data: dict, theme: str, mobile: bool, animated: bool) -> str:
    c = THEMES[theme]
    puzzle, solution, passes = data["puzzle"], data["solution"], data["passes"]
    sizes = data["decoding"]["level_sizes"]
    blanks = [k for k in range(81) if puzzle[k] == "0"]
    tiles = [("PUZZLE", f"{data['givens']} givens|{len(blanks)} blanks", puzzle, None)]
    for j, board in enumerate(passes):
        ok = sum(board[k] == solution[k] for k in blanks)
        code = "1 code" if sizes[j] == 1 else f"{sizes[j]} codes"
        tiles.append((f"PASS {j + 1} OF {len(passes)}", f"{code}|{ok}/{len(blanks)} blanks right", board, ok))
    w = 420 if mobile else 1080
    m = 28 if mobile else 32
    cols = 3 if mobile else 7
    cell = 11.5 if mobile else 15.5
    colw = (w - 2 * m) / cols
    rowh = cell * 9 + 82
    rows = -(-len(tiles) // cols)
    top = 150 if mobile else 124
    cap = ([f"{data['model'][0].upper()}{data['model'][1:]}.",
            (f"Board {data['board_index']} of the upstream {data['board_set'].split()[1]} test set, greedy decoding, "
            f"checkpoint SHA-256 {data['checkpoint']['generator_sha256'][:12]}."),
            "Each grid decodes the codes known after that pass; boxed cells differ from the solution. "
            "One board chosen by a fixed rule: " + data["board_rule"] + "."])
    cap_lines = [ln for t in cap for ln in wrap(t, 50 if mobile else 135)]
    h = int(top + rows * rowh + 30 + len(cap_lines) * (20 if mobile else 22) + 20)
    desc = (f"A hard Sudoku board solved by an independently trained MLX Next-Concept Modeling model in "
            f"{len(passes)} passes. After pass 1 the grid is mostly wrong; after pass {len(passes)} it matches the solution.")
    style = f"<style>{animation_css(len(tiles), 14, 0.9)}</style>" if animated else ""
    out = head(w, h, data["title"], desc, style, c)
    if mobile:
        out.append(f'<text x="{m}" y="52" font-size="26" font-weight="650">{escape(data["title"])}</text>')
        for j, line in enumerate(wrap("REAL OUTPUT · hard board · coarse to fine · MLX", 34)):
            out.append(f'<text x="{m}" y="{84 + j * 20}" font-size="15" fill="{c["muted"]}">{escape(line)}</text>')
    else:
        out.append(f'<text x="{m}" y="54" font-size="30" font-weight="650">{escape(data["title"])}</text>')
        out.append(f'<text x="{m}" y="84" font-size="18" fill="{c["muted"]}">REAL OUTPUT · hard board · '
                   f'one level per forward pass, coarse to fine · MLX</text>')
    for i, (label, sub, board, _ok) in enumerate(tiles):
        r, k = divmod(i, cols)
        x, y = m + k * colw, top + r * rowh
        out.append(f'<g class="r{i}">')
        out.append(f'<text x="{x}" y="{y - 8}" font-size="{12 if mobile else 12.5}" fill="{c["muted"]}" '
                   f'letter-spacing="0.5">{label}</text>')
        out.append(sudoku_tile(x, y, cell, board, puzzle, solution, c))
        if i:
            out.append(level_dots(x, y + 9 * cell + 8, 5 if mobile else 6, 2, i - 1, len(passes), c))
        for j, line in enumerate(sub.split("|")):
            out.append(f'<text x="{x}" y="{y + 9 * cell + 34 + j * 14}" font-size="{11 if mobile else 11.5}" '
                       f'fill="{c["muted"]}">{escape(line)}</text>')
        out.append("</g>")
    cy = top + rows * rowh + 4
    out.append(rule(m, w - m, cy, c))
    caption_lines(out, cap_lines, m, cy + 28, 14 if mobile else 15, 20 if mobile else 22, c)
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


def story_svg(data: dict, theme: str, mobile: bool, animated: bool) -> str:
    c = THEMES[theme]
    passes, sizes = data["passes"], data["decoding"]["level_sizes"]
    final = " ".join(passes[-1].split())
    first = final.split(". ")[0]
    sentence = first if first.endswith((".", "!", "?")) else first + "."
    rows = [(k, sizes[k], excerpt(passes[k], len(sentence))) for k in range(len(passes))]
    w = 420 if mobile else 1080
    m = 28 if mobile else 32
    row_h = 50 + 26 * max(len(wrap(t, 27)) for _, _, t in rows) if mobile else 56
    top = 150 if mobile else 134
    cap = [(f"{data['model'][0].upper()}{data['model'][1:]}; {data['evaluation']}. Unconditional TinyStories sample {data['decoding']['sample_index']} "
           f"of {data['decoding']['samples']}, seed {data['decoding']['seed']}, temperature {data['decoding']['temperature']:g}."),
           "Each row decodes the codes known after that pass and shows the first sentence. " + data["sample_rule"] + " "
           f"Checkpoint generator SHA-256 {data['checkpoint']['generator_sha256'][:12]}."]
    cap_lines = [ln for t in cap for ln in wrap(t, 50 if mobile else 135)]
    h = int(top + len(rows) * row_h + 10 + len(cap_lines) * (20 if mobile else 22) + 36)
    desc = (f"A TinyStories sample from an independently trained MLX NCM model sharpening over {len(rows)} passes; "
            f"the last pass reads: {rows[-1][2]}")
    style = f"<style>{animation_css(len(rows), 18, 1.2)}</style>" if animated else ""
    out = head(w, h, data["title"], desc, style, c)
    if mobile:
        out.append(f'<text x="{m}" y="52" font-size="26" font-weight="650">{escape(data["title"])}</text>')
        for j, line in enumerate(wrap(f"REAL OUTPUT · {data['model']} · 12 passes", 32)):
            out.append(f'<text x="{m}" y="{84 + j * 20}" font-size="15" fill="{c["muted"]}">{escape(line)}</text>')
    else:
        out.append(f'<text x="{m}" y="54" font-size="30" font-weight="650">{escape(data["title"])}</text>')
        out.append(f'<text x="{m}" y="84" font-size="18" fill="{c["muted"]}">REAL OUTPUT · {escape(data["model"])}, '
                   'not the released checkpoint · 12 passes</text>')
    prev = None
    for i, (lvl, codes, text) in enumerate(rows):
        y = top + i * row_h
        label = f"PASS {lvl + 1} OF {len(rows)} · " + ("1 code" if codes == 1 else f"{codes} codes")
        out.append(f'<g class="r{i}">')
        out.append(rule(m, w - m, y - 26, c))
        out.append(f'<text x="{m}" y="{y - 2}" font-size="15" fill="{c["muted"]}">{label}</text>')
        out.append(level_dots(m, y + 6, 10, 6, lvl, len(rows), c))
        if mobile:
            idx, ref_words = 0, (prev or "").split(" ") if prev else None
            for j, line in enumerate(wrap(text, 27)):
                words = line.split(" ")
                ref = " ".join(ref_words[idx : idx + len(words)]) if ref_words else None
                out.append(f'<text x="{m}" y="{y + 38 + j * 26}" font-size="21">{words_markup(ref, line, c["ink"], c["muted"])}</text>')
                idx += len(words)
        else:
            out.append(f'<text x="{m + 270}" y="{y + 8}" font-size="23">{words_markup(prev, text, c["ink"], c["muted"])}</text>')
        out.append("</g>")
        prev = text
    cy = top + len(rows) * row_h - 18
    out.append(rule(m, w - m, cy, c))
    caption_lines(out, cap_lines, m, cy + 28, 14 if mobile else 15, 20 if mobile else 22, c)
    out.append("</g></svg>")
    return "\n".join(out) + "\n"


SPLIT_SLOW, SPLIT_FAST, SPLIT_LEAD, SPLIT_HOLD = 0.55, 0.042, 0.9, 2.4


def split_times(slow_calls: int, total: int) -> list[float]:
    """Start time of each call 0..total: the first ``slow_calls`` calls are slow, the rest fast."""
    t, out = SPLIT_LEAD, [0.0]
    for n in range(1, total + 1):
        out.append(t)
        t += SPLIT_SLOW if n <= slow_calls else SPLIT_FAST
    out.append(t)
    return out


def split_css(times: list[float], cycle: float) -> tuple[str, dict]:
    """Keyframes that show an element from call a until call b (b None: until the loop ends)."""
    rules, names = [], {}

    def pct(t: float) -> str:
        return f"{100 * t / cycle:.3f}"

    def window(a: int, b: int | None) -> str:
        key = f"w{a}" if b is None else f"w{a}-{b}"
        if key not in names:
            name = key.replace("-", "_")
            on, off = "{visibility:visible}", "{visibility:hidden}"
            start = f"0%{on}" if a == 0 else f"0%{off}{pct(times[a])}%{on}"
            frames = start + ("" if b is None else f"{pct(times[b])}%,100%{off}")
            rules.append(f"@keyframes {name}{{{frames}}}.{name}{{animation-name:{name}}}")
            names[key] = name
        return names[key]

    rules.append(f"[class^=w]{{animation-duration:{cycle:.2f}s;animation-timing-function:step-end;"
                 "animation-iteration-count:infinite}")
    return rules, window


def display_pieces(raw: list[str]) -> list[str]:
    """Token pieces with whitespace runs collapsed to one space and no space at the start of the text."""
    out, space = [], True
    for p in raw:
        p = re.sub(r"\s+", " ", p)
        if space:
            p = p.lstrip()
        out.append(p)
        if p:
            space = p.endswith(" ")
    return out


_HELVETICA = dict(zip(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
    [278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278] + [556] * 10
    + [278, 278, 584, 584, 584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722,
       778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, 333, 556, 556, 500,
       556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500,
       500, 500, 334, 260, 334, 584]))
# Inter, the first font in FONT, sets wider than Helvetica, and changed tokens are semibold: measure every
# piece as if it were bold Inter so that a line never overflows and every level wraps by the same rule.
_BOLD_INTER = 1.16


def text_width(text: str, size: float) -> float:
    return sum(_HELVETICA.get(ch, 600) for ch in text) / 1000 * size * _BOLD_INTER


def wrap_pieces(pieces: list[str], width: float, size: float) -> list[list[tuple[int, str]]]:
    """Greedy line breaks between tokens that start with a space; returns (token index, text) per line."""
    lines, line, n = [], [], 0.0
    for i, p in enumerate(pieces):
        if p.startswith(" ") and line and n + text_width(p.rstrip(), size) > width:
            lines.append(line)
            line, n, p = [], 0.0, p.lstrip()
        line.append((i, p))
        n += text_width(p, size)
    if line:
        lines.append(line)
    return lines


def tail_words(text: str, limit: int) -> str:
    """The end of ``text`` cut at a word boundary to at most ``limit`` characters."""
    return ("…" + text[-limit:].split(" ", 1)[-1]) if len(text) > limit else text


def split_svg(data: dict, theme: str, mobile: bool, animated: bool) -> str:
    """The same text in both panels: NCM (left/top) sharpens all positions once per call, the AR (right/bottom)
    replays it one token per call."""
    c = THEMES[theme]
    ncm, ar, txt = data["ncm"], data["ar"], data["text"]
    sizes, calls_at = ncm["level_sizes"], ncm["calls_at_level"]
    levels, done = len(sizes), calls_at[-1]
    total, length = txt["window"], txt["length"]
    final = display_pieces(txt["pieces"])
    level_pieces = [display_pieces(p) for p in ncm["pass_pieces"]]
    toks = ncm["pass_tokens"]
    changed = [[True] * total] + [[a != b for a, b in zip(toks[k], toks[k - 1])] for k in range(1, levels)]
    w = 420 if mobile else 1080
    m = 24 if mobile else 32
    gap = 24
    pw = w - 2 * m if mobile else (w - 2 * m - gap) // 2
    pad = 16 if mobile else 22
    fs, lead = (14, 20) if mobile else (16, 23.5)
    tw = pw - 2 * pad
    strip_label = 13 if mobile else 15

    ncm_lines = [wrap_pieces(p, tw, fs) for p in level_pieces]
    final_lines = wrap_pieces(final, tw, fs)
    head_h = 176
    n_lines = max(len(final_lines), *(len(x) for x in ncm_lines))
    ph = int(head_h + (n_lines - 1) * lead + 58)

    times = split_times(done, total)
    cycle = round(times[-1] + SPLIT_HOLD, 2)
    rules, win = split_css(times, cycle)

    def cls(a: int, b: int | None) -> str:
        """Class attribute for an element shown from call a to call b; static figures show the final state."""
        if not animated:
            return ' visibility="hidden"' if b is not None and b <= total else ""
        return f' class="{win(a, b)}"' + (' visibility="hidden"' if b is not None else "")

    sub = f"REAL NCM SAMPLE · {data['model']} · the same {total} tokens in both panels"
    if mobile:
        title_lines = wrap(data["title"], 24)
        sub_lines = wrap(sub, 40)
        top = 70 + 30 * (len(title_lines) - 1) + 20 * len(sub_lines) + (70 if animated else 28)
    else:
        top = 172 if animated else 128
    ncm_xy = (m, top)
    ar_xy = (m, top + ph + 20) if mobile else (m + pw + gap, top)
    panels_end = (ar_xy[1] + ph) if mobile else top + ph

    sha = f"NCM generator {ncm['checkpoint']['generator_sha256'][:12]}, AR {ar['checkpoint']['weights_sha256'][:12]}"
    ar_size = f"{round(ar['checkpoint']['parameters'] / 1e6)}M"
    level0 = (f"NCM level 1 of {levels} is drawn from a stored unigram without a call, so its {levels} levels take "
              f"{done} calls. " if calls_at[0] == 0 else "")
    speed = f"The counter speeds up after call {done}." if animated else ""
    ended = (f"shown in full: all {total} tokens of the window" if length == total else
             f"the story ends at token {length} of the {total}-token window, and the text is shown to that point")
    d = data["decoding"]
    cap = [(f"Real NCM output of {data['model']}: unconditional TinyStories sample {ncm['sample_index']} of "
            f"{d['samples']}, seed {d['seed']}, temperature {d['temperature']:g}, {ended}."),
           (f"AR panel: our {ar_size} autoregressive baseline (same blocks, same training windows) replays the same "
            f"{total} tokens one call per token "
            "(forced decoding) to compare model calls for identical output; it is not an independent AR sample."),
           ("Model calls are counted, not timed: an NCM call is larger than an AR call, and NCM adds a codec "
            f"decode. {level0}Bold marks tokens changed by the latest NCM level. {speed}"),
           ("Whitespace is collapsed to single spaces in both panels. The first NCM sample of the seed with a "
            f"well-formed first sentence. SHA-256 {sha}.")]
    if not animated:
        strip_calls = [n for n in (0, 1, 3, 6, done, 32, 64, total) if n <= total]
        strip_h = 46 + len(strip_calls) * (52 if mobile else 30)
    else:
        strip_calls, strip_h = [], 0
    cap_lines = [ln for t in cap for ln in wrap(t.strip(), 56 if mobile else 135)]
    cy = panels_end + 36 + strip_h
    h = int(cy + 28 + len(cap_lines) * (20 if mobile else 22) + 12)

    def ncm_at(n: int) -> int:
        return max(k for k in range(levels) if calls_at[k] <= n)

    def flat(pieces: list[str], n: int) -> str:
        return " ".join("".join(pieces[:n]).split())

    def level_head(k: int) -> str:
        return " ".join("".join(level_pieces[k]).split())

    desc = (f"Split screen with a shared model-call counter and the same {total}-token TinyStories text in both "
            f"panels. Left, the NCM writes all {total} positions from the first level and sharpens them level by "
            f"level, done after {done} calls; its first level reads: {excerpt(level_head(0), 90)} Right, the "
            f"autoregressive model replays the same tokens one per call (forced decoding, not an independent "
            f"sample) and needs {total} calls; after {done} calls it has: {flat(final, done)} Both end with: "
            f"{excerpt(flat(final, total), 120)}")
    style = ""
    out_body: list[str] = []
    o = out_body.append

    if mobile:
        for j, line in enumerate(title_lines):
            o(f'<text x="{m}" y="{50 + j * 30}" font-size="26" font-weight="650">{escape(line)}</text>')
        y0 = 50 + 30 * (len(title_lines) - 1) + 32
        for j, line in enumerate(sub_lines):
            o(f'<text x="{m}" y="{y0 + j * 20}" font-size="15" fill="{c["muted"]}">{escape(line)}</text>')
    else:
        o(f'<text x="{m}" y="54" font-size="30" font-weight="650">{escape(data["title"])}</text>')
        o(f'<text x="{m}" y="84" font-size="18" fill="{c["muted"]}">{escape(sub)}</text>')

    if animated:
        cyc = top - (36 if mobile else 40)
        o(f'<text x="{m}" y="{cyc}" font-size="{20 if mobile else 22}" fill="{c["muted"]}">model calls:</text>')
        nx = m + (122 if mobile else 134)
        for n in range(total + 1):
            o(f'<text x="{nx}" y="{cyc}" font-size="{20 if mobile else 22}" font-weight="650"'
              f'{cls(n, n + 1 if n < total else None)}>{n}</text>')
        if not mobile:
            o(f'<text x="{w - m}" y="{cyc}" font-size="15" fill="{c["muted"]}" text-anchor="end">'
              'counted, not timed · one shared counter for both panels</text>')

    pitch = tw / total
    cw = max(pitch - (1.0 if mobile else 1.2), 1.2)

    def strip(x: float, y: float, filled: list[bool]) -> str:
        """Token positions as one path per colour (ink: filled, rule: empty)."""
        out = ""
        for on in (False, True):
            d = "".join(f"M{x + i * pitch:.2f} {y}h{cw:.2f}v12h-{cw:.2f}z" for i, f in enumerate(filled) if f == on)
            if d:
                out += f'<path d="{d}" fill="{c["ink"] if on else c["rule"]}"/>'
        return out

    def panel(xy, head: str, sub: str) -> tuple[float, float]:
        x, y = xy
        o(f'<rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="4" fill="{c["panel"]}"/>')
        o(f'<text x="{x + pad}" y="{y + 40}" font-size="{20 if mobile else 22}" font-weight="600">{escape(head)}</text>')
        o(f'<text x="{x + pad}" y="{y + 66}" font-size="{13 if mobile else 15}" fill="{c["muted"]}">{escape(sub)}</text>')
        return x + pad, y

    def body(x: float, y: float, lines, attr) -> None:
        """Text lines of token tspans; ``attr(i)`` gives each token's attributes."""
        for j, line in enumerate(lines):
            spans = "".join(f"<tspan{attr(i)}>{escape(p)}</tspan>" for i, p in line if p)
            o(f'<text x="{x}" y="{y + head_h + j * lead:.1f}" font-size="{fs}" xml:space="preserve">{spans}</text>')

    # NCM panel
    x, y = panel(ncm_xy, "NCM · one level per call", f"{levels} levels, coarse to fine, all {total} positions each time")
    dot, dgap = (7, 3) if mobile else (10, 4)
    for k in range(0 if animated else levels - 1, levels):
        a = calls_at[k]
        b = calls_at[k + 1] if k + 1 < levels else None
        o(f"<g{cls(a, b)}>")
        codes = "1 code" if sizes[k] == 1 else f"{sizes[k]} codes"
        lvl = f"LEVEL {k + 1}/{levels}" if mobile else f"LEVEL {k + 1} OF {levels}"
        o(f'<text x="{x}" y="{y + 102}" font-size="{strip_label}" fill="{c["muted"]}">{lvl} · {codes}</text>')
        o(level_dots(x + tw - levels * dot - (levels - 1) * dgap, y + 92 + (10 - dot), dot, dgap, k, levels, c))
        o("</g>")
        if k < levels - 1:
            o(f"<g{cls(a, b)}>")
            o(strip(x, y + 116, changed[k]))
            note = (f"level 1: stored unigram, no call · all {total} positions written" if k == 0 and calls_at[0] == 0
                    else "all positions written" if k == 0 else f"{sum(changed[k])} positions changed by this level")
            o(f'<text x="{x}" y="{y + 148}" font-size="{13 if mobile else 13.5}" fill="{c["muted"]}">{note}</text>')
            body(x, y, ncm_lines[k], lambda i, k=k: (f' fill="{c["ink"]}" font-weight="650"' if changed[k][i]
                                                     else f' fill="{c["muted"]}"'))
            o("</g>")
        else:
            if animated:
                o(f"<g{cls(a, a + 1)}>")
                o(strip(x, y + 116, changed[k]))
                o(f'<text x="{x}" y="{y + 148}" font-size="{13 if mobile else 13.5}" fill="{c["muted"]}">'
                  f"{sum(changed[k])} positions changed by this level</text>")
                body(x, y, ncm_lines[k], lambda i, k=k: (f' fill="{c["ink"]}" font-weight="650"' if changed[k][i]
                                                         else f' fill="{c["muted"]}"'))
                o("</g>")
            o(f"<g{cls(a + 1, None)}>" if animated else "<g>")
            o(strip(x, y + 116, [True] * total))
            o(f'<text x="{x}" y="{y + 148}" font-size="{13 if mobile else 13.5}" fill="{c["muted"]}">'
              f"complete: all {total} positions final</text>")
            body(x, y, final_lines, lambda i: "")
            o("</g>")
    o(f'<g{cls(done, None)}>' if animated else "<g>")
    o(f'<rect x="{x}" y="{y + ph - 34}" width="10" height="10" rx="1.5" fill="{c["ink"]}"/>')
    o(f'<text x="{x + 18}" y="{y + ph - 24}" font-size="{13 if mobile else 15}" fill="{c["muted"]}">'
      f'done after {done} calls</text></g>')

    # AR panel
    x, y = panel(ar_xy, "AR · one token per call",
                 f"{ar_size} AR baseline, forced decoding of the same text")
    o(f'<text x="{x}" y="{y + 102}" font-size="{strip_label}" fill="{c["muted"]}">TOKEN POSITIONS 1–{total}</text>')
    if animated:
        o(strip(x, y + 116, [False] * total))
    o("".join(f'<rect x="{x + i * pitch:.2f}" y="{y + 116}" width="{cw:.2f}" height="12" fill="{c["ink"]}"'
              f"{cls(i + 1, None)}/>" for i in range(total)))
    o(f'<text x="{x}" y="{y + 148}" font-size="{13 if mobile else 13.5}" fill="{c["muted"]}">'
      "positions written so far, one per call</text>")
    body(x, y, final_lines, lambda i: cls(i + 1, None))
    o(f'<g{cls(total, None)}>' if animated else "<g>")
    o(f'<rect x="{x}" y="{y + ph - 34}" width="10" height="10" rx="1.5" fill="{c["ink"]}"/>')
    end = f"done after {total} calls · the same text"
    o(f'<text x="{x + 18}" y="{y + ph - 24}" font-size="{13 if mobile else 15}" fill="{c["muted"]}">{end}</text></g>')

    # static: what each panel holds after n calls
    if strip_calls:
        sy = panels_end + 44
        o(f'<text x="{m}" y="{sy}" font-size="15" fill="{c["muted"]}" letter-spacing="0.5">'
          f'{"AFTER N MODEL CALLS" if mobile else "MODEL CALLS"}</text>')
        if not mobile:
            o(f'<text x="{m + 140}" y="{sy}" font-size="15" fill="{c["muted"]}">NCM</text>')
            o(f'<text x="{ar_xy[0]}" y="{sy}" font-size="15" fill="{c["muted"]}">AR</text>')
        lim = 30 if mobile else 36
        for i, n in enumerate(strip_calls):
            ry = sy + 30 + i * (52 if mobile else 30)
            k = ncm_at(n)
            if n < done:
                ncm_cell = f"level {k + 1}/{levels}: {excerpt(level_head(k), lim - 4)}"
            elif n == done:
                ncm_cell = f"level {levels}/{levels}: complete, all {total} tokens"
            else:
                ncm_cell = f"complete since call {done}"
            arw = flat(final, n)
            ar_cell = (f"{n}/{total} tokens: " + (tail_words(arw, lim - 12) if arw else "line breaks only" if n else "nothing yet")
                       if n < total else f"{total}/{total} tokens: the same text")
            o(rule(m, w - m, ry - 20, c))
            label = f"{n} CALL{'S' if n != 1 else ''}"
            muted = f' fill="{c["muted"]}"' if n > done else ""
            if mobile:
                o(f'<text x="{m}" y="{ry}" font-size="13" fill="{c["muted"]}">{label}</text>')
                o(f'<text x="{m + 76}" y="{ry}" font-size="13.5"><tspan fill="{c["muted"]}">NCM </tspan>'
                  f'<tspan{muted}>{escape(ncm_cell)}</tspan></text>')
                o(f'<text x="{m + 76}" y="{ry + 20}" font-size="13.5"><tspan fill="{c["muted"]}">AR </tspan>'
                  f"{escape(ar_cell)}</text>")
            else:
                o(f'<text x="{m}" y="{ry}" font-size="14" fill="{c["muted"]}">{label}</text>')
                o(f'<text x="{m + 140}" y="{ry}" font-size="15"{muted}>{escape(ncm_cell)}</text>')
                o(f'<text x="{ar_xy[0]}" y="{ry}" font-size="15">{escape(ar_cell)}</text>')

    o(rule(m, w - m, cy, c))
    caption_lines(out_body, cap_lines, m, cy + 28, 14 if mobile else 15, 20 if mobile else 22, c)

    if animated:
        fade = (f"@keyframes loop{{0%,{100 * (cycle - 0.5) / cycle:.2f}%{{opacity:1}}{100 * (cycle - 0.1) / cycle:.2f}%,"
                f"100%{{opacity:0}}}}.loop{{animation:loop {cycle:.2f}s linear infinite}}")
        style = "<style>@media (prefers-reduced-motion: no-preference){" + "".join(rules) + fade + "}</style>"
    out = head(w, h, data["title"], desc, style, c)
    out.append('<g class="loop">' if animated else "<g>")
    out.extend(out_body)
    out.append("</g></g></svg>")
    return "\n".join(out) + "\n"


def render(data: dict, name: str, out_dir: Path) -> list[Path]:
    draw = {"sudoku": sudoku_svg, "story": story_svg, "split": split_svg}[data["kind"]]
    written = []
    for animated in (False, True):
        stem = f"{name}-animated" if animated else name
        for theme in THEMES:
            for mobile in (False, True):
                path = out_dir / f"{stem}{'-mobile' if mobile else ''}-{theme}.svg"
                path.write_text(draw(data, theme, mobile, animated))
                written.append(path)
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)
    names = {"sudoku": "sudoku-12-passes", "story": "story-12-passes-87m", "split": "level-vs-token-87m"}
    for kind in ("sudoku", "story", "split"):
        sp = sub.add_parser(kind)
        sp.add_argument("--name", default=names[kind], help="file stem; use a new one when a figure changes")
        sp.add_argument("--out-dir", default=str(ASSETS))
        sp.add_argument("--render-only", action="store_true", help="redraw from NAME.json without running a model")
    s = sub.choices["sudoku"]
    s.add_argument("--config", default="configs/sudoku-e19-funded.json")
    s.add_argument("--weights", help="generator.safetensors (default: <config out_dir>/generator.safetensors)")
    s.add_argument("--codec", help="codec.safetensors (default: the config's codec_checkpoint)")
    s.add_argument("--boards", default="artifacts/upstream-sudoku")
    s.add_argument("--board-set", default="hard", choices=["easy", "medium", "hard"])
    s.add_argument("--board", type=int, help="board index; default: fewest givens among solved boards")
    t = sub.choices["story"]
    t.add_argument("--weights", help="export directory of a trained or converted text checkpoint (required unless --render-only)")
    t.add_argument("--label", default="independently trained 87M MLX NCM")
    t.add_argument("--eval-note", default="GPT-2 large generative perplexity 24.40")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n", type=int, default=16)
    t.add_argument("--sample", type=int, default=0)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--sample-rule", default="Sample 0 of the seed, shown without selection.")
    sp = sub.choices["split"]
    sp.add_argument("--ncm", help="NCM export directory (generator, codec, inference, bpe, spec)")
    sp.add_argument("--ar", help="ar.safetensors of the autoregressive baseline that replays the NCM text")
    sp.add_argument("--ar-config", help="the AR's JSON record with its config (default: --ar with .json)")
    sp.add_argument("--label", default="an independently trained 87M MLX NCM")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--n", type=int, default=16)
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--source-label", help="replaces the checkpoint directories in the JSON, e.g. '<export of run C>'")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    record = out_dir / f"{args.name}.json"
    if args.render_only:
        data = json.loads(record.read_text())
    else:
        if args.kind == "story" and not args.weights:
            p.error("story needs --weights unless --render-only")
        if args.kind == "split" and not (args.ncm and args.ar):
            p.error("split needs --ncm and --ar unless --render-only")
        data = {"sudoku": record_sudoku, "story": record_story, "split": record_split}[args.kind](args)
        if args.kind == "split" and args.source_label:
            text = json.dumps(data)
            for d in {str(Path(args.ncm)), str(Path(args.ar).parent)}:
                text = text.replace(json.dumps(d)[1:-1], args.source_label)
            data = json.loads(text)
        record.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    for path in render(data, args.name, out_dir):
        print(path)
    print(record)


if __name__ == "__main__":
    main()
