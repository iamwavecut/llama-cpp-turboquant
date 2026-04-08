#!/usr/bin/env python3

"""
Run a SpectralQuant benchmark matrix on top of llama-bench.

Paper alignment with the upstream SpectralQuant reference:
  - spectral_profile=nonuniform ~ "SQ-noQJL" style KV path
  - spectral_profile=selcorr   ~ "SQ-selQJL" style KV path

This script focuses on throughput and memory-budget comparisons using the
bench fields exposed by this fork:
  - model_size
  - context_size
  - compute_size
  - runtime_size
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    weights_mode: str
    cache_mode: str
    type_k: str
    type_v: str
    spectral_weight_calibration: str = ""
    spectral_kv_calibration: str = ""
    spectral_weight_profile: str = ""
    spectral_kv_profile: str = ""


@dataclass
class Result:
    scenario: str
    weights_mode: str
    cache_mode: str
    bench_kind: str
    model: str
    type_k: str
    type_v: str
    spectral_profile: str
    model_size: int
    context_size: int
    compute_size: int
    runtime_size: int
    avg_ts: float
    avg_ns: int
    stddev_ts: float
    stddev_ns: int
    throughput_delta_pct: float | None = None
    runtime_delta_pct: float | None = None
    context_delta_pct: float | None = None


def spectral_profile_display(scenario: Scenario) -> str:
    if scenario.spectral_weight_profile == scenario.spectral_kv_profile:
        return scenario.spectral_weight_profile
    if not scenario.spectral_weight_profile:
        return f"kv:{scenario.spectral_kv_profile}"
    if not scenario.spectral_kv_profile:
        return f"weight:{scenario.spectral_weight_profile}"
    return f"weight:{scenario.spectral_weight_profile};kv:{scenario.spectral_kv_profile}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SpectralQuant benchmark matrix via llama-bench")
    parser.add_argument("--bench-bin", default="build/bin/llama-bench", help="path to llama-bench")
    parser.add_argument("--legacy-model", required=True, help="baseline model path")
    parser.add_argument("--spectral-model", default="", help="optional SQ* weight model path")
    parser.add_argument("--spectral-calibration", default="", help="legacy alias: use the same spectral sidecar for SQ and SKV")
    parser.add_argument("--spectral-weight-calibration", default="", help="optional SQ spectral calibration sidecar")
    parser.add_argument("--spectral-kv-calibration", default="", help="optional SKV spectral calibration sidecar")
    parser.add_argument("--spectral-weight-profile", default="nonuniform", choices=("auto", "all", "nonuniform", "selcorr"), help="SQ runtime spectral profile")
    parser.add_argument("--spectral-kv-profiles", default="nonuniform,selcorr", help="comma-separated SKV profiles to benchmark")
    parser.add_argument(
        "--cache-type",
        default="skv4_0",
        choices=("skv2_0", "skv3_0", "skv4_0"),
        help="spectral KV type to benchmark",
    )
    parser.add_argument("--prompt", type=int, default=256, help="prompt tokens for pp benchmark")
    parser.add_argument("--gen", type=int, default=128, help="generated tokens for tg benchmark")
    parser.add_argument("--batch", type=int, default=512, help="batch size")
    parser.add_argument("--ubatch", type=int, default=128, help="ubatch size")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads")
    parser.add_argument("--reps", type=int, default=3, help="repetitions")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers for model loading")
    parser.add_argument("--no-kv-offload", type=int, choices=(0, 1), default=1, help="disable KV offload (default: 1)")
    parser.add_argument("--flash-attn", type=int, choices=(0, 1), default=1, help="enable flash attention (default: 1)")
    parser.add_argument("--use-mmap", type=int, choices=(0, 1), default=1, help="enable mmap (default: 1)")
    parser.add_argument(
        "--require-context-reduction",
        action="store_true",
        help="fail if any spectral KV scenario does not reduce context_size vs legacy for the same weights mode",
    )
    parser.add_argument("--output-json", default="", help="optional path to save raw JSON results")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra llama-bench argument; can be repeated")
    args = parser.parse_args()

    bench_bin = Path(args.bench_bin)
    if not bench_bin.exists():
        parser.error(f"llama-bench not found: {bench_bin}")
    if not Path(args.legacy_model).exists():
        parser.error(f"legacy model not found: {args.legacy_model}")
    if args.spectral_model and not Path(args.spectral_model).exists():
        parser.error(f"spectral model not found: {args.spectral_model}")
    if args.spectral_calibration:
        if args.spectral_weight_calibration or args.spectral_kv_calibration:
            parser.error("--spectral-calibration cannot be combined with the split weight/KV calibration flags")
        args.spectral_weight_calibration = args.spectral_calibration
        args.spectral_kv_calibration = args.spectral_calibration
    if args.spectral_weight_calibration and not Path(args.spectral_weight_calibration).exists():
        parser.error(f"spectral weight calibration not found: {args.spectral_weight_calibration}")
    if args.spectral_kv_calibration and not Path(args.spectral_kv_calibration).exists():
        parser.error(f"spectral KV calibration not found: {args.spectral_kv_calibration}")
    if args.spectral_kv_calibration and args.no_kv_offload != 1:
        parser.error("spectral KV scenarios require --no-kv-offload 1 because SKV* is CPU-only")
    if args.spectral_model and args.n_gpu_layers != 0:
        parser.error("spectral weight scenarios require --n-gpu-layers 0 because SQ* is CPU-only")

    kv_profiles = [profile.strip() for profile in args.spectral_kv_profiles.split(",") if profile.strip()]
    if not kv_profiles:
        parser.error("--spectral-kv-profiles produced an empty profile set")
    for profile in kv_profiles:
        if profile not in {"auto", "all", "nonuniform", "selcorr"}:
            parser.error(f"invalid KV profile: {profile}")
    args.spectral_kv_profiles = kv_profiles

    return args


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    scenarios = [
        Scenario(
            name="legacy_kv_fp_weights",
            model=args.legacy_model,
            weights_mode="fp",
            cache_mode="legacy",
            type_k="f16",
            type_v="f16",
        ),
    ]

    if args.spectral_kv_calibration:
        for profile in args.spectral_kv_profiles:
            scenarios.append(
                Scenario(
                    name=f"spectral_kv_{profile}_fp_weights",
                    model=args.legacy_model,
                    weights_mode="fp",
                    cache_mode="spectral",
                    type_k=args.cache_type,
                    type_v=args.cache_type,
                    spectral_kv_calibration=args.spectral_kv_calibration,
                    spectral_kv_profile=profile,
                )
            )

    if args.spectral_model:
        scenarios.append(
            Scenario(
                name="legacy_kv_spectral_weights",
                model=args.spectral_model,
                weights_mode="spectral",
                cache_mode="legacy",
                type_k="f16",
                type_v="f16",
                spectral_weight_calibration=args.spectral_weight_calibration,
                spectral_weight_profile=args.spectral_weight_profile,
            )
        )

        if args.spectral_kv_calibration:
            for profile in args.spectral_kv_profiles:
                scenarios.append(
                    Scenario(
                        name=f"spectral_kv_{profile}_spectral_weights",
                        model=args.spectral_model,
                        weights_mode="spectral",
                        cache_mode="spectral",
                        type_k=args.cache_type,
                        type_v=args.cache_type,
                        spectral_weight_calibration=args.spectral_weight_calibration,
                        spectral_kv_calibration=args.spectral_kv_calibration,
                        spectral_weight_profile=args.spectral_weight_profile,
                        spectral_kv_profile=profile,
                    )
                )

    return scenarios


def build_command(args: argparse.Namespace, scenario: Scenario, bench_kind: str) -> list[str]:
    cmd = [
        str(Path(args.bench_bin)),
        "-m", scenario.model,
        "-o", "jsonl",
        "-oe", "none",
        "--no-warmup",
        "-r", str(args.reps),
        "-t", str(args.threads),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "-ngl", str(args.n_gpu_layers),
        "-ctk", scenario.type_k,
        "-ctv", scenario.type_v,
        "-nkvo", str(args.no_kv_offload),
        "-fa", str(args.flash_attn),
        "-mmp", str(args.use_mmap),
    ]

    if bench_kind == "pp":
        cmd.extend(["-p", str(args.prompt), "-n", "0"])
    elif bench_kind == "tg":
        cmd.extend(["-p", "0", "-n", str(args.gen)])
    else:
        raise ValueError(f"unsupported bench kind: {bench_kind}")

    if scenario.spectral_weight_calibration:
        cmd.extend(["--spectral-weight-calibration", scenario.spectral_weight_calibration])
    if scenario.spectral_kv_calibration:
        cmd.extend(["--spectral-kv-calibration", scenario.spectral_kv_calibration])
    if scenario.spectral_weight_profile:
        cmd.extend(["--spectral-weight-profile", scenario.spectral_weight_profile])
    if scenario.spectral_kv_profile:
        cmd.extend(["--spectral-kv-profile", scenario.spectral_kv_profile])

    for extra in args.extra_arg:
        cmd.append(extra)

    return cmd


def run_bench(args: argparse.Namespace, scenario: Scenario, bench_kind: str) -> Result:
    cmd = build_command(args, scenario, bench_kind)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"llama-bench failed for {scenario.name}/{bench_kind}")

    records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(records) != 1:
        raise RuntimeError(f"expected exactly one JSONL record for {scenario.name}/{bench_kind}, got {len(records)}")

    rec = records[0]
    return Result(
        scenario=scenario.name,
        weights_mode=scenario.weights_mode,
        cache_mode=scenario.cache_mode,
        bench_kind=bench_kind,
        model=scenario.model,
        type_k=scenario.type_k,
        type_v=scenario.type_v,
        spectral_profile=spectral_profile_display(scenario),
        model_size=int(rec["model_size"]),
        context_size=int(rec["context_size"]),
        compute_size=int(rec["compute_size"]),
        runtime_size=int(rec["runtime_size"]),
        avg_ts=float(rec["avg_ts"]),
        avg_ns=int(rec["avg_ns"]),
        stddev_ts=float(rec["stddev_ts"]),
        stddev_ns=int(rec["stddev_ns"]),
    )


def annotate_deltas(results: list[Result]) -> None:
    baselines: dict[tuple[str, str], Result] = {}
    for result in results:
        if result.cache_mode == "legacy":
            baselines[(result.weights_mode, result.bench_kind)] = result

    for result in results:
        baseline = baselines.get((result.weights_mode, result.bench_kind))
        if baseline is None:
            continue
        result.throughput_delta_pct = 100.0 * (result.avg_ts / baseline.avg_ts - 1.0)
        result.runtime_delta_pct = 100.0 * (result.runtime_size / baseline.runtime_size - 1.0)
        result.context_delta_pct = 100.0 * (result.context_size / baseline.context_size - 1.0)


def validate_results(args: argparse.Namespace, results: list[Result]) -> None:
    if not args.require_context_reduction:
        return

    for result in results:
        if result.cache_mode != "spectral":
            continue
        if result.context_delta_pct is None or result.context_delta_pct >= 0.0:
            raise RuntimeError(
                f"context_size did not decrease for {result.scenario}/{result.bench_kind}: "
                f"context_delta_pct={result.context_delta_pct}"
            )


def format_bytes_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.3f}"


def format_bytes_mib(value: int) -> str:
    return f"{value / (1024 ** 2):.2f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def print_summary(results: Iterable[Result]) -> None:
    print("| 🧪 Сценарий | Режим | Весы | KV | Профиль | t/s | Runtime GiB | Context MiB | Compute MiB | Δ t/s vs legacy | Δ runtime vs legacy |")
    print("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            "| {scenario} | {bench_kind} | {weights_mode} | {cache_mode} | {profile} | {ts:.2f} | {runtime} | {context} | {compute} | {d_ts} | {d_runtime} |".format(
                scenario=result.scenario,
                bench_kind=result.bench_kind,
                weights_mode=result.weights_mode,
                cache_mode=result.cache_mode,
                profile=result.spectral_profile or "-",
                ts=result.avg_ts,
                runtime=format_bytes_gib(result.runtime_size),
                context=format_bytes_mib(result.context_size),
                compute=format_bytes_mib(result.compute_size),
                d_ts=format_delta(result.throughput_delta_pct),
                d_runtime=format_delta(result.runtime_delta_pct),
            )
        )


def main() -> int:
    args = parse_args()
    scenarios = build_scenarios(args)

    results: list[Result] = []
    for scenario in scenarios:
        for bench_kind in ("pp", "tg"):
            results.append(run_bench(args, scenario, bench_kind))

    annotate_deltas(results)
    validate_results(args, results)
    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bench_bin": str(Path(args.bench_bin)),
            "legacy_model": args.legacy_model,
            "spectral_model": args.spectral_model,
            "spectral_calibration": args.spectral_calibration,
            "cache_type": args.cache_type,
            "prompt": args.prompt,
            "gen": args.gen,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "threads": args.threads,
            "reps": args.reps,
            "results": [asdict(result) for result in results],
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
