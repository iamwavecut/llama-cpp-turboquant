#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


TIMINGS_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")
RSS_RE = re.compile(r"^\s*([0-9]+)\s+maximum resident set size\s*$", re.MULTILINE)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str


@dataclass(frozen=True)
class CacheSpec:
    name: str
    type_k: str | None
    type_v: str | None
    spectral: bool
    flash_attn: str


@dataclass
class Result:
    model: str
    weights_path: str
    cache_mode: str
    prompt_name: str
    prompt_file: str
    ctx_size: int
    ttft_ms: float
    pp_tps: float
    tg_tps: float
    model_size: int
    max_rss_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemma4 Q8/SQ3/SQ4 with legacy KV and SKV3")
    parser.add_argument("--cli-bin", default="build/bin/llama-cli")
    parser.add_argument("--bench-bin", default="build/bin/llama-bench")
    parser.add_argument(
        "--q8-model",
        default="models/gemma4-96e-a4b-heretic/Gemma-4-96E-A4B-Heretic-Q8_0.gguf",
    )
    parser.add_argument(
        "--sq3-model",
        default="models/gemma4-96e-a4b-heretic/Gemma-4-96E-A4B-Heretic-SQ3_1S.gguf",
    )
    parser.add_argument(
        "--sq4-model",
        default="models/gemma4-96e-a4b-heretic/Gemma-4-96E-A4B-Heretic-SQ4_1S.gguf",
    )
    parser.add_argument(
        "--short-prompt",
        default="models/gemma4-96e-a4b-heretic/bench_prompt_short.txt",
    )
    parser.add_argument(
        "--full-prompt",
        default="models/gemma4-96e-a4b-heretic/bench_prompt_full.txt",
    )
    parser.add_argument("--spectral-calibration", required=True)
    parser.add_argument("--ctx-size-short", type=int, default=1024)
    parser.add_argument("--ctx-size-full", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--ubatch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--only-models",
        default="",
        help="comma-separated subset from: q8_0,sq3_1s,sq4_1s",
    )
    args = parser.parse_args()

    for path in (
        args.cli_bin,
        args.bench_bin,
        args.q8_model,
        args.short_prompt,
        args.full_prompt,
        args.spectral_calibration,
    ):
        if not Path(path).exists():
            parser.error(f"missing path: {path}")

    return args


def run_cli_once(
    args: argparse.Namespace,
    model: ModelSpec,
    cache: CacheSpec,
    prompt_name: str,
    prompt_file: str,
    ctx_size: int,
) -> tuple[float, float, float, int]:
    cmd = [
        "/usr/bin/time",
        "-l",
        args.cli_bin,
        "-m", model.path,
        "-dev", "none",
        "-ngl", "0",
        "-t", str(args.threads),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-c", str(ctx_size),
        "-n", str(args.n_predict),
        "-f", prompt_file,
        "-nkvo",
        "-fa", cache.flash_attn,
        "-st",
        "--simple-io",
        "--show-timings",
        "--no-display-prompt",
        "--reasoning", "off",
        "--reasoning-format", "none",
        "--temp", "0",
        "--repeat-penalty", "1.0",
        "--log-disable",
    ]
    if cache.type_k is not None:
        cmd.extend(["-ctk", cache.type_k])
    if cache.type_v is not None:
        cmd.extend(["-ctv", cache.type_v])
    if cache.spectral:
        cmd.extend(["--spectral-calibration", args.spectral_calibration, "--spectral-profile", "auto"])

    started = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    first_byte_at = None
    output = bytearray()

    while True:
        chunk = proc.stdout.read(1)
        if not chunk:
            break
        if first_byte_at is None:
            first_byte_at = time.perf_counter()
        output.extend(chunk)

    proc.wait()
    stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
    stdout_text = output.decode("utf-8", errors="replace")
    text = stdout_text + "\n" + stderr_text
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed for {model.name}/{cache.name}/{prompt_name}:\n{text}")

    match = TIMINGS_RE.search(text)
    if not match:
        raise RuntimeError(f"failed to parse timings for {model.name}/{cache.name}/{prompt_name}:\n{text}")

    rss_match = RSS_RE.search(stderr_text)
    max_rss_bytes = int(rss_match.group(1)) if rss_match else 0
    ttft_ms = ((first_byte_at or time.perf_counter()) - started) * 1000.0
    return ttft_ms, float(match.group(1)), float(match.group(2)), max_rss_bytes


def main() -> int:
    args = parse_args()

    selected = {item.strip().lower() for item in args.only_models.split(",") if item.strip()}
    available_models = [
        ("q8_0", ModelSpec("Q8_0", args.q8_model)),
        ("sq3_1s", ModelSpec("SQ3_1S", args.sq3_model)),
        ("sq4_1s", ModelSpec("SQ4_1S", args.sq4_model)),
    ]
    models = []
    for key, spec in available_models:
        if selected and key not in selected:
            continue
        if not Path(spec.path).exists():
            continue
        models.append(spec)

    if not models:
        raise RuntimeError("no benchmarkable models were selected or found")
    caches = [
        CacheSpec("orig_ctx", None, None, False, "off"),
        CacheSpec("skv3_ctx", "skv3_0", "skv3_0", True, "on"),
    ]
    prompts = [
        ("short", args.short_prompt, args.ctx_size_short),
        ("full", args.full_prompt, args.ctx_size_full),
    ]

    results: list[Result] = []

    for model in models:
        for cache in caches:
            for prompt_name, prompt_file, ctx_size in prompts:
                print(f"[bench] start model={model.name} cache={cache.name} prompt={prompt_name} ctx={ctx_size}", flush=True)
                ttft_values = []
                pp_values = []
                tg_values = []
                rss_values = []

                for _ in range(args.reps):
                    ttft_ms, pp_tps, tg_tps, max_rss_bytes = run_cli_once(
                        args,
                        model,
                        cache,
                        prompt_name,
                        prompt_file,
                        ctx_size,
                    )
                    ttft_values.append(ttft_ms)
                    pp_values.append(pp_tps)
                    tg_values.append(tg_tps)
                    rss_values.append(max_rss_bytes)

                results.append(Result(
                    model=model.name,
                    weights_path=model.path,
                    cache_mode=cache.name,
                    prompt_name=prompt_name,
                    prompt_file=prompt_file,
                    ctx_size=ctx_size,
                    ttft_ms=sum(ttft_values) / len(ttft_values),
                    pp_tps=sum(pp_values) / len(pp_values),
                    tg_tps=sum(tg_values) / len(tg_values),
                    model_size=Path(model.path).stat().st_size,
                    max_rss_bytes=max(rss_values),
                ))
                print(
                    f"[bench] done model={model.name} cache={cache.name} prompt={prompt_name} "
                    f"ttft_ms={results[-1].ttft_ms:.1f} pp_tps={results[-1].pp_tps:.2f} "
                    f"tg_tps={results[-1].tg_tps:.2f} rss_gib={results[-1].max_rss_bytes / (1024 ** 3):.3f}",
                    flush=True,
                )
                if args.output_json:
                    Path(args.output_json).write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")

    headers = (
        "model",
        "cache",
        "prompt",
        "ctx",
        "ttft_ms",
        "pp_tps",
        "tg_tps",
        "model_gib",
        "rss_gib",
    )
    print("\t".join(headers))
    for result in results:
        print("\t".join((
            result.model,
            result.cache_mode,
            result.prompt_name,
            str(result.ctx_size),
            f"{result.ttft_ms:.1f}",
            f"{result.pp_tps:.2f}",
            f"{result.tg_tps:.2f}",
            f"{result.model_size / (1024 ** 3):.3f}",
            f"{result.max_rss_bytes / (1024 ** 3):.3f}",
        )))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
