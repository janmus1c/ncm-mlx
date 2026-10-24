"""Error attribution and per-level temperature on a trained checkpoint.

    python examples/level_ablation.py --config configs/toy-grammar-12k.json

Part 1 keeps the codec's true codes for levels 0..k of held-out sentences and
samples the remaining levels; validity per k shows where parallel sampling
loses accuracy. Part 2 samples unconditionally with lower temperature at the
fine levels only.
"""

import argparse
import json

import mlx.core as mx
import numpy as np

from ncm_mlx.cli import _load
from ncm_mlx.config import Config
from ncm_mlx.eval.metrics import sample_report
from ncm_mlx.generation.sampler import SampleOptions, generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/toy-grammar-12k.json")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--out")
    args = p.parse_args()
    codec, gen, corpus = _load(Config.load(args.config))
    tok = corpus.tokenizer
    K = codec.vq.num_levels
    val = mx.array(corpus.val[: args.n])
    true_codes = codec.encode(val)

    def decode(g):
        return [tok.decode(r) for r in np.array(g.tokens).tolist()]

    forced = {}
    for k in range(K + 1):
        texts = []
        for seed in range(2):
            g = generate(gen, codec, val.shape[0], SampleOptions(temperature=1.0, seed=seed), forced=true_codes[:k])
            texts += decode(g)
        r = sample_report(texts, corpus.train_text, corpus.val_text)
        exact = np.mean([t.rstrip("_") in set(corpus.val_text[: args.n]) for t in texts])
        forced[f"true_levels_0..{k - 1}" if k else "none"] = {"valid": r["valid"], "in_val": round(float(exact), 4)}

    schedules = {
        "T=1.0 all": [1.0] * K,
        "T=0.7 all": [0.7] * K,
        "coarse 1.0, levels>=2 0.5": [1.0, 1.0] + [0.5] * (K - 2),
        "coarse 1.0, levels>=2 0.0": [1.0, 1.0] + [0.0] * (K - 2),
        "coarse 1.0, levels>=3 0.0": [1.0, 1.0, 1.0] + [0.0] * (K - 3),
    }
    temps = {}
    for name, sched in schedules.items():
        texts = []
        for seed in range(2):
            texts += decode(generate(gen, codec, 512, SampleOptions(seed=seed, level_temperatures=sched)))
        temps[name] = sample_report(texts, corpus.train_text, corpus.val_text)

    result = {"forced_prefix": forced, "level_temperatures": temps}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            f.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
