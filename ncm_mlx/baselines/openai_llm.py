"""Score a chat LLM behind any OpenAI-compatible endpoint on the exact Sudoku task.

    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=... \\
        python -m ncm_mlx.baselines.openai_llm --model <id> --level easy --n 50

This is a reference point for quality and latency, not a matched comparison:
the LLM is orders of magnitude larger, was not trained on this task and answers
one puzzle per request. Thinking is disabled by default (the
``enable_thinking`` chat-template flag) so the latency is that of a direct
answer; ``--think`` allows reasoning tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

from ncm_mlx.text import upstream_sudoku

PROMPT = """Solve this Sudoku. Dots are blanks. Every row, column and 3x3 box must contain the digits 1-9 once.
Keep every given digit. Reply with exactly 9 lines of 9 digits and nothing else.

{grid}"""


def grid_text(puzzle: np.ndarray) -> str:
    cells = "".join("." if c == 0 else str(int(c)) for c in puzzle)
    return "\n".join(cells[r * 9 : r * 9 + 9] for r in range(9))


def parse_grid(text: str) -> np.ndarray | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    digits = re.findall(r"[1-9]", "".join(re.findall(r"^[\s1-9|,.-]*$", text, flags=re.MULTILINE)))
    if len(digits) < 81:
        return None
    return np.array([int(d) for d in digits[-81:]], dtype=np.int32)


def ask(base_url: str, key: str, model: str, prompt: str, max_tokens: int, timeout: float,
        think: bool = False) -> tuple[str, dict]:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    msg = out["choices"][0]["message"]
    return msg.get("content") or "", {**out.get("usage", {}), "finish_reason": out["choices"][0].get("finish_reason"), "reasoning_chars": len(msg.get("reasoning_content") or msg.get("reasoning") or "")}


def main() -> int:
    from ncm_mlx.eval.metrics import sudoku_report

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--boards", default="artifacts/upstream-sudoku")
    p.add_argument("--think", action="store_true")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out")
    args = p.parse_args()
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
    key = os.environ.get("OPENAI_API_KEY", "")

    puzzles, solutions = upstream_sudoku.load(Path(args.boards), args.level)
    puzzles, solutions = puzzles[: args.n], solutions[: args.n]
    preds, latencies, tokens, unparsed = [], [], [], 0
    for puzzle in puzzles:
        prompt = PROMPT.format(grid=grid_text(puzzle))
        t0 = time.perf_counter()
        text, usage = ask(base_url, key, args.model, prompt, args.max_tokens, args.timeout, args.think)
        latencies.append(time.perf_counter() - t0)
        tokens.append(usage.get("completion_tokens", 0))
        grid = parse_grid(text)
        if grid is None:
            unparsed += 1
            grid = np.zeros(81, dtype=np.int32)
        preds.append(grid)
    report = sudoku_report(puzzles, np.stack(preds), solutions)
    result = {
        "model": args.model, "level": args.level, "think": args.think, "n": len(puzzles),
        "scores": report, "unparsed": unparsed,
        "latency_s_p50": round(statistics.median(latencies), 2),
        "latency_s_mean": round(statistics.mean(latencies), 2),
        "completion_tokens_mean": round(statistics.mean(tokens), 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
