"""Compare XSum summarization in MLX with upstream PyTorch on CPU, on fixed inputs.

Both sides run in one process with identical prefixes. The report covers:

1. GPT-2 tokenizer: ids and decoded text against Hugging Face ``transformers``.
2. Per-level generator logits with teacher-forced codes, in float32 and (both sides exact) float64.
3. Free-running greedy generation: codes at every level, decoder logits, the penalized token
   readout and the decoded summaries, using the released defaults (temperature 1, top-k 1,
   top-p 0, which is greedy) and classifier-free guidance at scale 1.0.

The readout and the text cleanup of the reference side are upstream's own functions: they are
taken from ``scripts/eval_shard.py`` of the checkout by parsing the file and executing only the
``pick_tokens`` and ``dec`` definitions, so nothing is copied here. Needs PyTorch and upstream's
dependencies for this run only::

    uv run --with torch --with omegaconf --with hydra-core --with regex --with transformers \\
        python scripts/official_xsum_parity.py --upstream /path/to/HazyResearch-ncm

``--fixture`` writes the small upstream-side reference used by ``tests/test_official_summarize.py``.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TEXTS = [
    ("The council said on Tuesday that it would close three libraries next year, saving about £1.2m. "
    "Residents have objected, arguing that the buildings are used by hundreds of children every week."),
    "Short text.",
    "Café owners in München won't open on Sundays — \"it's not worth it,\" one said.\n\nNumbers: 3.14, 12,000 and x2.",
    "Police are appealing for witnesses after a car was set alight overnight. " * 70,
]
TOKENIZER_CASES = TEXTS + ["  leading and trailing  ", "tabs\tand\nnewlines\n\n  x", "I'm we've you're it's I'd we'll",
                           "a<|endoftext|>b", "emoji \U0001f600 naïve", "'quoted' ''double'' end'", "!!! ??? ..."]


def upstream_extract(upstream: Path, names: tuple[str, ...], namespace: dict) -> dict:
    """Execute only the named top-level functions of upstream's ``eval_shard.py`` in ``namespace``."""
    tree = ast.parse((upstream / "scripts" / "eval_shard.py").read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in body} != set(names):
        raise SystemExit("eval_shard.py no longer defines " + ", ".join(names))
    exec(compile(ast.Module(body, []), "eval_shard.py", "exec"), namespace)  # noqa: S102
    return namespace


def run_upstream(net, prefix_t, forced=None, temperature=0.0, top_k=0, top_p=1.0, cfg=0.0, eos=50256):
    """Upstream ``generate`` with every level's raw logits recorded; ``forced`` replaces the sampled codes."""
    import torch

    seen = []
    original = net._sample_level

    def hook(logits, t, k, p):
        seen.append(logits.detach().clone())
        if forced is not None:
            return forced[len(seen) - 1]
        return original(logits, t, k, p)

    net._sample_level = hook
    try:
        with torch.no_grad():
            out = net.generate(None, prefix_t, temperature=temperature, top_k=top_k, top_p=top_p,
                               cfg_scale=cfg, cfg_null_eos=eos)
    finally:
        del net._sample_level
    return out, seen


def run(args) -> dict:
    import mlx.core as mx
    import torch
    import transformers
    from official_parity import build_upstream, exact_float64_ops, max_abs, rope64, split_levels
    from omegaconf import OmegaConf

    from ncm_mlx.official import sample, spec, summarize

    mx.set_default_device(mx.cpu)
    mx.set_memory_limit(16 * 2**30)
    torch.manual_seed(0)
    task = spec.get("xsum")
    src = Path(args.src or ROOT / "artifacts/official/xsum")
    cfg, vqvae, net, _, inference = build_upstream(Path(args.upstream), src)
    m = sample.load(src / "mlx")
    sys.path.insert(0, str(Path(args.upstream)))
    from tokenization.hf_tokenizer import HFTokenizer

    special = {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"}
    hf = HFTokenizer(cfg.tokenizer.pre_tokenizer.model_id, **special)
    hf_pinned = HFTokenizer("openai-community/gpt2", revision=spec.GPT2_REVISION, **special)
    gen_cfg = cfg.generation
    P, levels = task.generator.prefix_len, list(task.codec.levels)
    EOS, PAD = hf.eos_token_id, hf.pad_token_id
    report = {"task": "xsum", "torch": torch.__version__, "transformers": transformers.__version__,
              "mlx": mx.__version__, "device": "cpu", "upstream_code_revision": spec.UPSTREAM_CODE_REVISION,
              "checkpoint_revision": spec.REVISION, "dataset_revision": spec.DATASET_REVISION,
              "gpt2_revision": spec.GPT2_REVISION}
    assert (EOS, PAD) == (task.eos_id, task.pad_id)
    assert OmegaConf.to_container(gen_cfg)["top_k"] == task.decoding.top_k

    # 1. tokenizer
    cases = TOKENIZER_CASES
    articles, refs = summarize.load_test_set(src / "data")
    ids_equal = [m.bpe.encode(c) == hf.hf_tokenizer(c)["input_ids"] for c in cases]
    ref_equal = [m.bpe.encode(r) == hf.hf_tokenizer(r)["input_ids"] for r in refs[:500]]
    sample_ids = [a[:60] for a in articles[:200]] + [list(range(0, 50304, 997))[:40]]
    dec_raw = [m.bpe.decode([t for t in s if t < m.bpe.vocab_size], skip_special=True) for s in sample_ids]
    dec_hf = [hf.decode(s, skip_special_tokens=True) for s in sample_ids]
    dec_clean = [summarize.clean_up_tokenization(t) for t in dec_raw]
    report["tokenizer"] = {
        "transformers_clean_up_tokenization_spaces_default": bool(hf.hf_tokenizer.clean_up_tokenization_spaces),
        "pinned_revision_equals_default": all(hf.hf_tokenizer(c)["input_ids"] == hf_pinned.hf_tokenizer(c)["input_ids"]
                                                for c in cases),
        "encode_cases": len(cases), "encode_equal": all(ids_equal),
        "encode_references_checked": 500, "encode_references_equal": sum(ref_equal),
        "decode_rows": len(sample_ids),
        "decode_equal_raw": sum(a == b for a, b in zip(dec_raw, dec_hf)),
        "decode_equal_after_clean_up": sum(a == b for a, b in zip(dec_clean, dec_hf)),
    }

    # 2. inputs: four test articles (dataset ids) and the four raw texts (Hugging Face tokenization)
    rows = [summarize.prefix_row(a, P, PAD) for a in articles[:4]]
    rows += [(hf.hf_tokenizer(t)["input_ids"][:P] + [PAD] * P)[:P] for t in TEXTS]
    mine = [summarize.prefix_row(summarize.tokenize_article(m, t), P, PAD) for t in TEXTS]
    report["inputs"] = {"rows": len(rows), "test_articles": [0, 1, 2, 3], "texts": len(TEXTS),
                        "our_prefix_ids_equal_for_texts": mine == rows[4:],
                        "padding_tokens_per_row": [int(sum(t == PAD for t in r)) for r in rows]}
    prefix_t = torch.tensor(rows)
    prefix_m = mx.array(np.array(rows, dtype=np.int32))

    # 3. free-running greedy through the released config path (temperature 1, top-k 1, top-p 0)
    out_cfg_t, logits_free = run_upstream(net, prefix_t, temperature=float(gen_cfg.temperature),
                                          top_k=int(gen_cfg.top_k), top_p=float(gen_cfg.top_p))
    out_greedy_t, _ = run_upstream(net, prefix_t, temperature=0.0)
    report["upstream_config_path_equals_argmax"] = bool(torch.equal(out_cfg_t, out_greedy_t))
    codes_t = split_levels(out_greedy_t.numpy(), levels)
    forced_t = [torch.tensor(c) for c in codes_t]
    forced_m = [mx.array(c.astype(np.int32)) for c in codes_t]

    g = sample.generate(m, prefix_m, temperature=task.decoding.temperature, top_k=task.decoding.top_k,
                        top_p=task.decoding.top_p, keep_logits=True)
    mlx_codes = [np.array(c) for c in g.codes]
    report["free_running_greedy"] = {
        "codes_equal_per_level": [bool(np.array_equal(a, b)) for a, b in zip(mlx_codes, codes_t)],
        "level_logits_max_abs_diff_while_codes_agree": [
            max_abs(lm, lt.numpy()) if np.array_equal(a, b) else None
            for lm, lt, a, b in zip(g.level_logits, logits_free, mlx_codes, codes_t)]}

    # 4. teacher-forced per-level logits, float32
    _, logits_t = run_upstream(net, prefix_t, forced=forced_t)
    gf = sample.generate(m, prefix_m, keep_logits=True, forced=forced_m)
    per_level = []
    for lt, lm in zip(logits_t, gf.level_logits):
        lt, lm = lt.numpy(), np.array(lm)
        per_level.append({"max_abs_diff": max_abs(lm, lt), "max_abs_logit": float(np.abs(lt).max()),
                          "argmax_equal": float((lm.argmax(-1) == lt.argmax(-1)).mean())})
    report["forced_level_logits_fp32"] = per_level
    report["forced_level_logits_fp32_max_abs_diff"] = max(r["max_abs_diff"] for r in per_level)
    nocache = sample.generate(m, prefix_m, keep_logits=True, forced=forced_m, prefix_cache=False)
    report["prefix_cache_vs_reference_path_max_abs_diff"] = max(
        max_abs(a, b) for a, b in zip(gf.level_logits, nocache.level_logits))

    # 5. float64 on both sides
    import models.common as up_common
    import models.vae as up_vae

    def rms64(self, x):
        import torch as _t
        return x * _t.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

    saved = (up_common.RMSNorm.forward, up_vae.RMSNorm.forward)
    up_common.RMSNorm.forward = up_vae.RMSNorm.forward = rms64
    torch.set_default_dtype(torch.float64)
    vq64 = copy.deepcopy(vqvae).double()
    net64 = copy.deepcopy(net).double()
    net64.vqvae_local = (vq64,)
    hd = task.generator.dim // task.generator.n_head
    for j, n in enumerate(levels):
        cos, sin = rope64(np.concatenate([np.arange(P), np.arange(n)]), hd, task.generator.rope_base)
        setattr(net64, f"rope_cos_gen_{j}", torch.tensor(cos))
        setattr(net64, f"rope_sin_gen_{j}", torch.tensor(sin))
    for head in net64.lm_heads:
        if hasattr(head, "E_sq"):
            head.E_sq = (head.E ** 2).sum(-1)
            head.d_scale = head.E_sq.mean().clamp(min=1e-12)
    _, logits64_t = run_upstream(net64, prefix_t, forced=forced_t)
    torch.set_default_dtype(torch.float32)
    up_common.RMSNorm.forward, up_vae.RMSNorm.forward = saved
    m.codec.astype(mx.float64)
    m.gen.astype(mx.float64)
    with exact_float64_ops():
        g64 = sample.generate(m, prefix_m, keep_logits=True, forced=forced_m)
    m.codec.astype(mx.float32)
    m.gen.astype(mx.float32)
    report["forced_level_logits_fp64_max_abs_diff"] = max(max_abs(a, b.numpy()) for a, b in zip(g64.level_logits, logits64_t))
    report["fp32_rounding_reference"] = {
        "torch_fp32_vs_torch_fp64": max(max_abs(a.numpy(), b.numpy()) for a, b in zip(logits_t, logits64_t)),
        "mlx_fp32_vs_torch_fp64": max(max_abs(a, b.numpy()) for a, b in zip(gf.level_logits, logits64_t))}

    # 6. decoder logits, penalized readout, text: upstream's own pick_tokens and dec
    allow = inference["token_unigram"].double()
    ns = {"torch": torch, "ALLOW": allow, "freq": float(gen_cfg.frequency_penalty),
          "mu": float(gen_cfg.frequency_deficit), "ADJ": [float(v) for v in str(gen_cfg.adjacency_penalty).split(",")],
          "CUT_AT_EOS": bool(gen_cfg.get("cut_at_eos", True)), "EOS": EOS, "PAD": PAD,
          "tk": SimpleNamespace(pre_tokenizer=hf)}
    ns = upstream_extract(Path(args.upstream), ("pick_tokens", "dec"), ns)
    with torch.no_grad():
        _, zl = vqvae.vq.idx_to_zhats(out_greedy_t, return_zlevels=True)
        dlog_t = vqvae.decoder(zl[-1])
        ids_t = ns["pick_tokens"](dlog_t).numpy()
    texts_t = [ns["dec"](r) for r in ids_t.tolist()]
    dlog_m = m.codec.decode_logits(g.final)
    ids_m = summarize.readout(m, dlog_m, task.decoding)
    texts_m = [m.decode(r) for r in ids_m]
    texts_m_clean = [summarize.clean_up_tokenization(t) for t in texts_m]
    plain_t = dlog_t.argmax(-1).numpy()
    report["readout"] = {
        "decoder_logits_max_abs_diff": max_abs(dlog_m, dlog_t.numpy()),
        "decoder_logits_max_abs": float(dlog_t.abs().max()),
        "penalized_ids_equal_rows": int((ids_m == ids_t).all(-1).sum()), "rows": len(rows),
        "penalized_ids_equal_given_upstream_decoder_logits": bool(np.array_equal(
            summarize.penalized_readout(dlog_t.numpy(), inference["token_unigram"].numpy(), task.decoding), ids_t)),
        "rows_where_penalty_changes_argmax": int((plain_t != ids_t).any(-1).sum()),
        "text_equal_raw_decode": [a == b for a, b in zip(texts_m, texts_t)],
        "text_equal_after_clean_up": [a == b for a, b in zip(texts_m_clean, texts_t)]}

    # 7. end-to-end through summarize_prefixes (what the CLI runs)
    s = summarize.summarize_prefixes(m, prefix_m)
    report["end_to_end"] = {"texts_equal_upstream": [a == b for a, b in zip(s.texts, texts_t)],
                            "ids_equal_upstream": bool(np.array_equal(s.token_ids, ids_t))}

    # 8. classifier-free guidance at scale 1.0
    out_cfg1_t, logits_cfg_t = run_upstream(net, prefix_t, temperature=0.0, cfg=1.0, eos=EOS)
    gc = sample.generate(m, prefix_m, temperature=0.0, keep_logits=True, cfg_scale=1.0)
    codes_c = split_levels(out_cfg1_t.numpy(), levels)
    report["cfg_scale_1"] = {
        "codes_equal_per_level": [bool(np.array_equal(np.array(a), b)) for a, b in zip(gc.codes, codes_c)],
        "differs_from_unguided": bool(not torch.equal(out_cfg1_t, out_greedy_t)),
        "raw_logits_max_abs_diff_per_level": [
            max_abs(a, b.numpy()) if np.array_equal(np.array(c), cc) else None
            for a, b, c, cc in zip(gc.level_logits, logits_cfg_t, gc.codes, codes_c)]}

    if args.fixture:
        k = 5
        fixture = {"tokenizer": {"encode": [[c, hf.hf_tokenizer(c)["input_ids"]] for c in cases]},
                   "rows": [r for r in rows[:k]], "codes": out_greedy_t[:k].tolist(),
                   "readout_ids": ids_t[:k].tolist(), "texts": texts_t[:k],
                   "cfg1_codes": out_cfg1_t[:k].tolist(),
                   "forced_logits": [{"row0_head": np.round(lt[0, :2, :16].numpy(), 5).tolist(),
                                      "row0_argmax": lt[0].argmax(-1).tolist()} for lt in logits_t],
                   "decoder_logits_row0_head": np.round(dlog_t[0, :3, :12].numpy(), 5).tolist(),
                   "meta": {k_: report[k_] for k_ in ("torch", "transformers", "upstream_code_revision",
                                                     "checkpoint_revision")}}
        Path(args.fixture).write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="checkout of HazyResearch/ncm")
    ap.add_argument("--src", help="release files (default artifacts/official/xsum)")
    ap.add_argument("--fixture", help="write the upstream-side reference JSON used by the tests")
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
