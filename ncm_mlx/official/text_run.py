"""``official eval-text``: one report comparing data, the released checkpoint, a trained NCM and its AR baseline."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ncm_mlx.official import text_eval as te
from ncm_mlx.official import tinystories as ts
from ncm_mlx.official.bpe import BPETokenizer
from ncm_mlx.official.train_config import TextConfig


def _decode(bpe: BPETokenizer, rows) -> list[str]:
    return [bpe.decode(te.cut(r), skip_special=True) for r in rows]


def run_text_eval(cfg: TextConfig, args, runtime: dict, log=print) -> dict:
    from ncm_mlx.official import trainer
    from ncm_mlx.official.ar import ar_report

    d = cfg.data
    directory = ts.split_dir(Path(d.cache_root), d.seq_len, d.limit)
    manifest = json.loads((directory / "manifest.json").read_text())
    test = np.asarray(ts.load_split(directory, "test")[: args.test_rows])
    train = ts.load_split(directory, "train")
    bpe = BPETokenizer.load(d.bpe)
    openings = te.train_opening_index(np.asarray(train))
    out_dir = Path(args.out).parent if args.out else Path("runs") / f"{cfg.name}-eval"
    report = {"config": cfg.to_dict(), "runtime": runtime, "data_manifest": manifest,
              "test_rows": len(test), "seeds": args.seeds, "samples_per_seed": args.n,
              "sampling": {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "level0": "stored unigram"},
              "gpu_before": te.gpu_utilization(), "started": te.stamp(), "models": {}}
    samplers = {}
    if "official" in args.models:
        m, run, fwd = te.ncm_sampler(Path(args.official_weights))
        samplers["official"] = (run, fwd, m)
    if "ncm" in args.models:
        m, run, fwd = te.ncm_sampler(trainer.export_dir(cfg))
        samplers["ncm"] = (run, fwd, m)
    if "ar" in args.models:
        model, run, fwd = te.ar_sampler(cfg)
        samplers["ar"] = (run, fwd, model)

    for name in args.models:
        log(f"{te.stamp()} evaluating {name}")
        entry: dict = {}
        if name == "data":
            k = args.n
            rows_by_seed = {s: test[i * k : (i + 1) * k].tolist() for i, s in enumerate(args.seeds)
                            if (i + 1) * k <= len(test)}
            entry["note"] = "held-out test windows, decoded with the same cleanup; disjoint per seed"
        else:
            run, fwd, model = samplers[name]
            if name in ("official", "ncm"):
                tf = trainer.TeacherForced(model.gen)
                entry["codec_test"] = trainer.codec_report(model.codec, test)
                entry["generator_test_code_ce"] = trainer.generator_report(tf, model.codec, test)
                entry["parameters"] = {
                    "codec": sum(v.size for _, v in _flat(model.codec)),
                    "generator": sum(v.size for _, v in _flat(model.gen))}
            else:
                entry["ar_test"] = ar_report(model, test)
                entry["parameters"] = {"ar": sum(v.size for _, v in _flat(model))}
            entry["model_forwards_per_sample"] = fwd
            rows_by_seed = te.draw(run, args.n, args.seeds, args.batch)
        texts_by_seed = {s: _decode(bpe, rows) for s, rows in rows_by_seed.items()}
        all_rows = [r for rows in rows_by_seed.values() for r in rows]
        all_texts = [t for texts in texts_by_seed.values() for t in texts]
        entry["diagnostics"] = te.diagnostics(all_rows, all_texts, bpe, openings)
        entry["examples_seed_first"] = texts_by_seed[next(iter(texts_by_seed))][:8]
        te.write_samples(out_dir / f"samples-{name}.jsonl", name, rows_by_seed, texts_by_seed)
        report["models"][name] = entry
        mx.clear_cache()

    if not args.no_scorer:
        log(f"{te.stamp()} scoring with {te.SCORER_REPO}")
        scorer = te.Scorer()
        for name in args.models:
            texts_by_seed: dict[int, list[str]] = {}
            for line in (out_dir / f"samples-{name}.jsonl").read_text().splitlines():
                r = json.loads(line)
                texts_by_seed.setdefault(r["seed"], []).append(r["text"])
            report["models"][name]["gen_ppl"] = te.gen_ppl(scorer, texts_by_seed)
            log(f"{name}: gen ppl {report['models'][name]['gen_ppl']['mean']}")
        del scorer
        mx.clear_cache()

    if not args.no_bench:
        report["gpu_before_bench"] = te.gpu_utilization()
        for name, (run, fwd, _) in samplers.items():
            log(f"{te.stamp()} latency {name}")
            report["models"][name]["latency"] = te.latency(run, d.seq_len, fwd)
        report["gpu_after_bench"] = te.gpu_utilization()
    report["finished"] = te.stamp()
    return report


def _flat(module):
    from mlx.utils import tree_flatten

    return tree_flatten(module.parameters())
