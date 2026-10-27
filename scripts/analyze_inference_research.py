"""Check finite-distribution identities and derive config-bound cache accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/research/inference-analysis.json"


def distribution(values):
    assert all(value >= 0 for value in values)
    assert sum(values) == 1
    return values


def correction(p, q):
    residual = [max(pi - qi, F(0)) for pi, qi in zip(p, q, strict=True)]
    mass = sum(residual)
    return [value / mass for value in residual] if mass else [F(0)] * len(p)


def check_one_step(p, q):
    distribution(p)
    distribution(q)
    accepted = [min(pi, qi) for pi, qi in zip(p, q, strict=True)]
    rejection = 1 - sum(accepted)
    output = [ai + rejection * ri for ai, ri in zip(accepted, correction(p, q), strict=True)]
    tv = sum(abs(pi - qi) for pi, qi in zip(p, q, strict=True)) / 2
    assert output == p
    assert sum(accepted) == 1 - tv


def two_token_output(p_first, p_second, q_first, q_second):
    size = len(p_first)
    output = {(a, b): F(0) for a, b in product(range(size), repeat=2)}
    first_residual = correction(p_first, q_first)
    for a, b in product(range(size), repeat=2):
        draft_mass = q_first[a] * q_second[b]
        if not draft_mass:
            continue
        accept_first = min(F(1), p_first[a] / q_first[a])
        for x, y in product(range(size), repeat=2):
            output[x, y] += draft_mass * (1 - accept_first) * first_residual[x] * p_second[x][y]
        accept_second = min(F(1), p_second[a][b] / q_second[b])
        output[a, b] += draft_mass * accept_first * accept_second
        for y, residual in enumerate(correction(p_second[a], q_second)):
            output[a, y] += draft_mass * accept_first * (1 - accept_second) * residual
    return output


def probability_checks():
    fixtures = [
        ([F(1, 2), F(1, 3), F(1, 6)], [F(1, 4), F(1, 2), F(1, 4)]),
        ([F(1, 2), F(1, 2), F(0)], [F(0), F(0), F(1)]),
        ([F(1, 2), F(1, 2), F(0)], [F(1), F(0), F(0)]),
        ([F(1, 2), F(1, 2), F(0)], [F(1, 2), F(1, 2), F(0)]),
    ]
    for p, q in fixtures:
        check_one_step(p, q)
    p_first = distribution([F(3, 5), F(2, 5)])
    p_second = [distribution([F(1, 4), F(3, 4)]), distribution([F(2, 3), F(1, 3)])]
    expected = {(a, b): p_first[a] * p_second[a][b] for a, b in product(range(2), repeat=2)}
    mixture = dict.fromkeys(expected, F(0))
    for weight, q_first, q_second in [
        (F(1, 3), [F(4, 5), F(1, 5)], [F(1, 3), F(2, 3)]),
        (F(2, 3), [F(1, 5), F(4, 5)], [F(3, 4), F(1, 4)]),
    ]:
        got = two_token_output(p_first, p_second, distribution(q_first), distribution(q_second))
        assert got == expected
        for pair, value in got.items():
            mixture[pair] += weight * value
    assert mixture == expected
    assert sum(F(1, 2) ** j for j in range(5)) == F(31, 16)
    return {
        "arithmetic": "exact rational enumeration, not Monte Carlo",
        "one_step_fixtures": len(fixtures),
        "latent_conditioned_two_token_fixtures": 2,
        "latent_mixture_matches_target_joint": True,
        "scope": "finite mathematical fixtures only; no model, kernel, cache or speed validation",
    }


def analyze():
    path = ROOT / "configs/sudoku-e20-seed1.json"
    raw = path.read_bytes()
    cfg = json.loads(raw)
    levels = cfg["codec"]["levels"]
    prefix = cfg["generator"]["prefix_len"]
    layers = cfg["generator"]["n_layer"]
    width = cfg["generator"]["n_embd"]
    count = len(levels)
    before_positions = count * prefix + sum(levels)
    after_positions = prefix + sum(levels)
    before_pairs = sum(prefix**2 + length * prefix + length**2 for length in levels)
    after_pairs = prefix**2 + sum(length * prefix + length**2 for length in levels)
    toy_path = ROOT / "configs/toy-grammar-12k.json"
    toy_raw = toy_path.read_bytes()
    toy = json.loads(toy_raw)
    toy_prefix = toy["generator"]["prefix_len"]
    toy_levels = toy["codec"]["levels"]
    return {
        "status": "analytic accounting and finite probability checks; no new model experiment",
        "config": path.relative_to(ROOT).as_posix(),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "attention": cfg["generator"]["attn_pattern"],
        "levels": levels,
        "prefix_positions": prefix,
        "sum_level_positions": sum(levels),
        "linear_token_work_per_transformer_layer": {
            "repeated_prefix": before_positions,
            "prefix_once": after_positions,
            "removed_positions": before_positions - after_positions,
        },
        "logical_allowed_attention_pairs_per_layer_per_head": {
            "repeated_prefix": before_pairs,
            "prefix_once": after_pairs,
            "scope": "mask-allowed pairs, not physical kernel operations or measured latency",
        },
        "retained_prefix_kv_payload": {
            "layers": layers,
            "width": width,
            "kv_heads_equal_query_heads": True,
            "fp32_bytes_per_sample": 2 * layers * prefix * width * 4,
            "fp16_bytes_per_sample": 2 * layers * prefix * width * 2,
            "scope": "logical K/V payload; excludes allocator capacity, temporaries and model weights",
        },
        "tinyworld_comparison": {
            "config": toy_path.relative_to(ROOT).as_posix(),
            "config_sha256": hashlib.sha256(toy_raw).hexdigest(),
            "prefix_positions": toy_prefix,
            "level_count": len(toy_levels),
            "repeated_prefix_positions_per_layer": len(toy_levels) * toy_prefix + sum(toy_levels),
            "prefix_once_positions_per_layer": toy_prefix + sum(toy_levels),
        },
        "probability_checks": probability_checks(),
        "assumptions": [
            "Deterministic evaluation mode with a fixed full prefix and unchanged model parameters.",
            "Build the prefix cache during the first concept-level forward; reuse it for later levels.",
            "Do not cache generated-level states across changing concept levels.",
            "No inference speed or accuracy forecast follows from these counts.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare recomputed analysis with committed JSON")
    args = parser.parse_args()
    result = analyze()
    if args.check:
        if result != json.loads(OUTPUT.read_text()):
            raise SystemExit("Inference analysis differs; review inputs and regenerate")
        print("Verified cache accounting and exact finite-distribution fixtures")
    else:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
