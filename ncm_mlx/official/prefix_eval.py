"""Evaluation of prompt continuation on TinyStories (``data.prefix_len > 0``).

The test prompts are the first ``P`` tokens of held-out stories; the reference
continuation is the next 128 tokens of the same story. Every model continues the
same prompts at the same sampling settings (temperature 1, no top-k/top-p).
Reported separately:

* Teacher-forced held-out loss of the continuation (NCM code cross-entropy per
  level, AR next-token cross-entropy), with the true prompt and with the prompt
  of the next test story (``*_shuffled_prompt``). The gap is how much the model
  reads its prompt.
* The sample diagnostics of ``text_eval`` on the continuations alone.
* Generative perplexity under the fixed GPT-2 large scorer. The scorer reads
  ``prompt + continuation``; only the GPT-2 tokens after the number of tokens the
  prompt alone encodes are counted. A token that straddles the boundary is
  therefore counted as continuation. Held-out gold continuations are scored the
  same way.
* A prompt-faithfulness **diagnostic**: name carry-over, content-word overlap and
  verbatim copying, each next to a control that pairs every continuation with the
  prompt of the next story. These are surface heuristics, not coherence or quality
  measures, and a high copy rate is a failure, not a success.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from ncm_mlx.official import text_eval as te
from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.bpe import BPETokenizer
from ncm_mlx.official.train_config import TextConfig

WORD = re.compile(r"[A-Za-z]+")
SENTENCE_END = set('.!?"\n')


def prompt_names(text: str) -> list[str]:
    """Capitalised words that do not start a sentence or line, in order of first appearance."""
    names: list[str] = []
    for m in WORD.finditer(text):
        w = m.group()
        if len(w) < 2 or not (w[0].isupper() and w[1:].islower()):
            continue
        before = text[: m.start()].rstrip(" \t")
        if not before or before[-1] in SENTENCE_END:
            continue
        if w not in names:
            names.append(w)
    return names


def content_words(text: str, min_len: int = 4) -> set[str]:
    return {w.lower() for w in WORD.findall(text) if len(w) >= min_len}


def _copied_ngram_share(prompt_ids: list[int], cont_ids: list[int], n: int = 4) -> float | None:
    grams = te._ngrams(cont_ids, n)
    if not grams:
        return None
    seen = set(te._ngrams(prompt_ids, n))
    return sum(g in seen for g in grams) / len(grams)


def faithfulness(prompts: list[str], prompt_ids: list[list[int]], conts: list[str],
                 cont_ids: list[list[int]], shift: int = 0) -> dict:
    """Surface overlap of each continuation with the prompt ``shift`` stories later (0 = its own)."""
    n = len(conts)
    mention, coverage, overlap, copied, new_caps = [], [], [], [], []
    for i in range(n):
        j = (i + shift) % n
        names = prompt_names(prompts[j])
        if names:
            hit = [bool(re.search(rf"\b{re.escape(w)}\b", conts[i])) for w in names]
            mention.append(any(hit))
            coverage.append(sum(hit) / len(names))
        words = content_words(conts[i])
        if words:
            overlap.append(len(words & content_words(prompts[j])) / len(words))
        share = _copied_ngram_share(prompt_ids[j], cont_ids[i])
        if share is not None:
            copied.append(share)
        new_caps.append(len({w for w in prompt_names(conts[i]) if w not in names}))

    def mean(xs, k):
        return round(k * float(np.mean(xs)), 3) if xs else None

    return {
        "samples_with_prompt_names": len(mention),
        "name_mention_pct": mean(mention, 100),
        "name_coverage_pct": mean(coverage, 100),
        "content_word_overlap_pct": mean(overlap, 100),
        "copied_4gram_pct": mean(copied, 100),
        "mean_new_capitalised_words": mean(new_caps, 1),
    }


def faithfulness_report(prompts, prompt_ids, conts, cont_ids) -> dict:
    return {"label": "diagnostic: surface overlap heuristics, not a quality measure",
            "own_prompt": faithfulness(prompts, prompt_ids, conts, cont_ids, 0),
            "control_next_story_prompt": faithfulness(prompts, prompt_ids, conts, cont_ids, 1)}


def _shuffled(rows: np.ndarray, P: int) -> np.ndarray:
    out = np.array(rows)
    out[:, :P] = np.roll(rows[:, :P], 1, axis=0)
    return out


def ncm_continuer(m, P: int):
    from ncm_mlx.official import sample

    if m.task.generator.prefix_len != P:
        raise SystemExit(f"checkpoint has prefix_len {m.task.generator.prefix_len}, the config asks for {P}")

    def run(prompts: np.ndarray, seed: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        return sample.generate(m, mx.array(prompts.astype(np.int32)), temperature=temperature, top_k=top_k,
                               top_p=top_p, seed=seed).tokens

    return run


def ar_continuer(model):
    def run(prompts: np.ndarray, seed: int, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
        return model.generate(len(prompts), temperature, top_k, top_p, seed, prefix=mx.array(prompts.astype(np.int32)))

    return run


def continue_prompts(run, prompts: np.ndarray, seeds: list[int], batch: int) -> dict[int, list[list[int]]]:
    out = {}
    for seed in seeds:
        rows = []
        for s in range(0, len(prompts), batch):
            rows += np.array(run(prompts[s : s + batch], seed * 100_003 + s)).tolist()
        out[seed] = rows
    return out


def run_prefix_eval(cfg: TextConfig, args, runtime: dict, log=print) -> dict:
    from ncm_mlx.official import ar as ar_mod
    from ncm_mlx.official import sample, trainer

    d, P = cfg.data, cfg.data.prefix_len
    directory = ts.split_dir(Path(d.cache_root), d.seq_len, d.limit, P)
    manifest = json.loads((directory / "manifest.json").read_text())
    test = np.asarray(ts.load_split(directory, "test")[: args.test_rows])
    train = ts.load_split(directory, "train")
    bpe = BPETokenizer.load(d.bpe)
    n = min(args.n, len(test))
    prompts, gold = test[:n, :P], test[:n, P:]
    prompt_ids = [[int(t) for t in r] for r in prompts]
    prompt_texts = [bpe.decode(r, skip_special=True) for r in prompt_ids]
    openings = te.train_opening_index(np.ascontiguousarray(np.asarray(train)[:, P:]))
    models = [m for m in args.models if m != "official"]
    if len(models) != len(args.models):
        log("official checkpoint skipped: it has a one-token prefix and cannot read a prompt")
    out_dir = Path(args.out).parent if args.out else Path("runs") / f"{cfg.name}-eval"
    report = {"config": cfg.to_dict(), "runtime": runtime, "data_manifest": manifest, "prefix_len": P,
              "test_rows": len(test), "prompts": n, "seeds": args.seeds,
              "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0,
                           "level0": "predicted from the prompt (learned queries)"},
              "scorer_protocol": "scorer reads prompt + continuation; only tokens after the prompt are counted",
              "started": te.stamp(), "models": {}}
    texts: dict[str, dict[int, list[str]]] = {}
    for name in models:
        log(f"{te.stamp()} evaluating {name}")
        entry: dict = {}
        shuffled = _shuffled(test, P)
        if name == "data":
            rows_by_seed = {args.seeds[0]: gold.tolist()}
            entry["note"] = "held-out gold continuations of the same prompts, same cleanup (one entry, no sampling)"
        elif name == "ncm":
            m = sample.load(trainer.export_dir(cfg))
            tf = trainer.TeacherForced(m.gen)
            entry["codec_test_target"] = trainer.codec_report(m.codec, test[:, P:])
            entry["generator_test_code_ce"] = trainer.generator_report(tf, m.codec, test, prefix_len=P)
            entry["generator_test_code_ce_shuffled_prompt"] = trainer.generator_report(
                tf, m.codec, shuffled, prefix_len=P)
            entry["model_forwards_per_sample"] = len(m.codec.levels)
            rows_by_seed = continue_prompts(ncm_continuer(m, P), prompts, args.seeds, args.batch)
        else:
            model = ar_mod.load_ar(cfg)
            entry["ar_test"] = ar_mod.ar_report(model, test)
            entry["ar_test_shuffled_prompt"] = ar_mod.ar_report(model, shuffled)
            entry["model_forwards_per_sample"] = P - 1 + model.seq_len
            rows_by_seed = continue_prompts(ar_continuer(model), prompts, args.seeds, args.batch)
        texts[name] = {s: [bpe.decode(te.cut(r), skip_special=True) for r in rows] for s, rows in rows_by_seed.items()}
        flat_rows = [r for rows in rows_by_seed.values() for r in rows]
        flat_texts = [t for ts_ in texts[name].values() for t in ts_]
        entry["diagnostics"] = te.diagnostics(flat_rows, flat_texts, bpe, openings)
        cut_ids = [te.cut(r) for r in flat_rows]
        k = len(rows_by_seed)
        entry["prompt_faithfulness"] = faithfulness_report(prompt_texts * k, prompt_ids * k, flat_texts, cut_ids)
        first = next(iter(rows_by_seed))
        entry["examples_seed_first"] = [{"prompt": p, "continuation": t}
                                        for p, t in zip(prompt_texts[:8], texts[name][first][:8])]
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"continuations-{name}.jsonl").open("w") as f:
            for seed, rows in rows_by_seed.items():
                for i, (r, t) in enumerate(zip(rows, texts[name][seed])):
                    f.write(json.dumps({"model": name, "seed": seed, "index": i, "prompt": prompt_texts[i],
                                        "continuation": t, "ids": r}) + "\n")
        report["models"][name] = entry
        mx.clear_cache()

    if not args.no_scorer:
        log(f"{te.stamp()} scoring with {te.SCORER_REPO}")
        scorer = te.Scorer()
        for name in models:
            report["models"][name]["gen_ppl_continuation"] = te.gen_ppl(scorer, texts[name], prompt_texts)
            log(f"{name}: continuation gen ppl {report['models'][name]['gen_ppl_continuation']['mean']}")
        del scorer
        mx.clear_cache()
    report["finished"] = te.stamp()
    return report
