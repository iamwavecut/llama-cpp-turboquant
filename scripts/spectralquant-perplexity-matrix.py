#!/usr/bin/env python3

"""
Run a SpectralQuant perplexity / KL-divergence matrix on top of llama-perplexity.

Paper alignment with the upstream SpectralQuant reference:
  - spectral_profile=nonuniform ~ "SQ-noQJL" style KV path
  - spectral_profile=selcorr   ~ "SQ-selQJL" style KV path

This is an acceptance harness for this GGUF runtime, not a promise of paper-exact
headline numbers from the upstream PyTorch prototype.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PLUS_MINUS_RE = r"(?:±|\+/-)"


FINAL_PPL_RE = re.compile(
    rf"Final estimate:\s*PPL\s*=\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)

MEAN_PPL_Q_RE = re.compile(
    rf"Mean PPL\(Q\)\s*:\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)
MEAN_PPL_BASE_RE = re.compile(
    rf"Mean PPL\(base\)\s*:\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)
MEAN_PPL_RATIO_RE = re.compile(
    rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)
MEAN_PPL_DIFF_RE = re.compile(
    rf"Mean PPL\(Q\)-PPL\(base\)\s*:\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)
MEAN_KLD_RE = re.compile(
    rf"Mean\s+KLD:\s*({FLOAT_RE})\s*{PLUS_MINUS_RE}\s*({FLOAT_RE})",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    model: str
    weights_mode: str
    cache_mode: str
    type_k: str
    type_v: str
    spectral_calibration: str = ""
    spectral_profile: str = ""


@dataclass
class Result:
    scenario: str
    weights_mode: str
    cache_mode: str
    model: str
    type_k: str
    type_v: str
    spectral_profile: str
    model_size: int
    ppl_value: float
    ppl_unc: float
    ppl_base_value: float | None = None
    ppl_base_unc: float | None = None
    ppl_ratio_value: float | None = None
    ppl_ratio_unc: float | None = None
    ppl_diff_value: float | None = None
    ppl_diff_unc: float | None = None
    mean_kld_value: float | None = None
    mean_kld_unc: float | None = None
    ppl_delta_vs_legacy_pct: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SpectralQuant perplexity matrix via llama-perplexity")
    parser.add_argument("--perplexity-bin", default="build/bin/llama-perplexity", help="path to llama-perplexity")
    parser.add_argument("--legacy-model", required=True, help="baseline FP/legacy model path")
    parser.add_argument("--spectral-model", default="", help="optional SQ* weight model path")
    parser.add_argument("--spectral-calibration", default="", help="optional SKV spectral calibration sidecar")
    parser.add_argument("--dataset", required=True, help="plain-text evaluation corpus")
    parser.add_argument(
        "--cache-type",
        default="skv4_0",
        choices=("skv2_0", "skv3_0", "skv4_0"),
        help="spectral KV type to evaluate",
    )
    parser.add_argument("--ctx-size", type=int, default=1024, help="perplexity context size")
    parser.add_argument("--batch", type=int, default=2048, help="batch size")
    parser.add_argument("--ubatch", type=int, default=512, help="ubatch size")
    parser.add_argument("--chunks", type=int, default=8, help="maximum number of chunks to process")
    parser.add_argument("--ppl-stride", type=int, default=0, help="optional strided perplexity mode")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers for model loading")
    parser.add_argument("--no-kv-offload", type=int, choices=(0, 1), default=1, help="disable KV offload (default: 1)")
    parser.add_argument(
        "--flash-attn",
        choices=("on", "off", "auto"),
        default="on",
        help="flash attention setting (default: on)",
    )
    parser.add_argument("--use-mmap", type=int, choices=(0, 1), default=1, help="enable mmap (default: 1)")
    parser.add_argument(
        "--require-kld-metrics",
        action="store_true",
        help="fail if any non-baseline spectral scenario does not emit KL/PPL ratio metrics",
    )
    parser.add_argument("--kl-base-file", default="", help="optional path to persist baseline logits")
    parser.add_argument("--output-json", default="", help="optional path to save parsed JSON results")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra llama-perplexity argument; can be repeated")
    args = parser.parse_args()

    perplexity_bin = Path(args.perplexity_bin)
    if not perplexity_bin.exists():
        parser.error(f"llama-perplexity not found: {perplexity_bin}")
    if not Path(args.legacy_model).exists():
        parser.error(f"legacy model not found: {args.legacy_model}")
    if args.spectral_model and not Path(args.spectral_model).exists():
        parser.error(f"spectral model not found: {args.spectral_model}")
    if not Path(args.dataset).exists():
        parser.error(f"dataset not found: {args.dataset}")
    if args.spectral_calibration and not Path(args.spectral_calibration).exists():
        parser.error(f"spectral calibration not found: {args.spectral_calibration}")
    if args.spectral_calibration and args.no_kv_offload != 1:
        parser.error("spectral KV scenarios require --no-kv-offload 1 because SKV* is CPU-only")
    if args.spectral_model and args.n_gpu_layers != 0:
        parser.error("spectral weight scenarios require --n-gpu-layers 0 because SQ* is CPU-only")

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

    if args.spectral_calibration:
        for profile in ("nonuniform", "selcorr"):
            scenarios.append(
                Scenario(
                    name=f"spectral_kv_{profile}_fp_weights",
                    model=args.legacy_model,
                    weights_mode="fp",
                    cache_mode="spectral",
                    type_k=args.cache_type,
                    type_v=args.cache_type,
                    spectral_calibration=args.spectral_calibration,
                    spectral_profile=profile,
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
            )
        )

        if args.spectral_calibration:
            for profile in ("nonuniform", "selcorr"):
                scenarios.append(
                    Scenario(
                        name=f"spectral_kv_{profile}_spectral_weights",
                        model=args.spectral_model,
                        weights_mode="spectral",
                        cache_mode="spectral",
                        type_k=args.cache_type,
                        type_v=args.cache_type,
                        spectral_calibration=args.spectral_calibration,
                        spectral_profile=profile,
                    )
                )

    return scenarios


def build_command(
    args: argparse.Namespace,
    scenario: Scenario,
    kl_base_file: str,
    use_kl_divergence: bool,
) -> list[str]:
    cmd = [
        str(Path(args.perplexity_bin)),
        "-m", scenario.model,
        "-f", args.dataset,
        "-c", str(args.ctx_size),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "--chunks", str(args.chunks),
        "-t", str(args.threads),
        "-ngl", str(args.n_gpu_layers),
        "-ctk", scenario.type_k,
        "-ctv", scenario.type_v,
        "--flash-attn", args.flash_attn,
        "--ppl-output-type", "0",
        "--kl-divergence-base", kl_base_file,
    ]

    if args.ppl_stride > 0:
        cmd.extend(["--ppl-stride", str(args.ppl_stride)])

    cmd.append("--no-kv-offload" if args.no_kv_offload else "--kv-offload")
    cmd.append("--mmap" if args.use_mmap else "--no-mmap")

    if scenario.spectral_calibration:
        cmd.extend(["--spectral-calibration", scenario.spectral_calibration])
    if scenario.spectral_profile:
        cmd.extend(["--spectral-profile", scenario.spectral_profile])
    if use_kl_divergence:
        cmd.append("--kl-divergence")

    for extra in args.extra_arg:
        cmd.append(extra)

    return cmd


def extract_float_pair(pattern: re.Pattern[str], text: str, metric_name: str) -> tuple[float, float]:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"failed to parse {metric_name}")
    return float(match.group(1)), float(match.group(2))


def parse_result(text: str, scenario: Scenario, model_size: int, use_kl_divergence: bool) -> Result:
    if not use_kl_divergence:
        ppl_value, ppl_unc = extract_float_pair(FINAL_PPL_RE, text, "baseline perplexity")
        return Result(
            scenario=scenario.name,
            weights_mode=scenario.weights_mode,
            cache_mode=scenario.cache_mode,
            model=scenario.model,
            type_k=scenario.type_k,
            type_v=scenario.type_v,
            spectral_profile=scenario.spectral_profile,
            model_size=model_size,
            ppl_value=ppl_value,
            ppl_unc=ppl_unc,
        )

    ppl_value, ppl_unc = extract_float_pair(MEAN_PPL_Q_RE, text, "Mean PPL(Q)")
    ppl_base_value, ppl_base_unc = extract_float_pair(MEAN_PPL_BASE_RE, text, "Mean PPL(base)")
    ppl_ratio_value, ppl_ratio_unc = extract_float_pair(MEAN_PPL_RATIO_RE, text, "Mean PPL(Q)/PPL(base)")
    ppl_diff_value, ppl_diff_unc = extract_float_pair(MEAN_PPL_DIFF_RE, text, "Mean PPL(Q)-PPL(base)")
    mean_kld_value, mean_kld_unc = extract_float_pair(MEAN_KLD_RE, text, "Mean KLD")

    return Result(
        scenario=scenario.name,
        weights_mode=scenario.weights_mode,
        cache_mode=scenario.cache_mode,
        model=scenario.model,
        type_k=scenario.type_k,
        type_v=scenario.type_v,
        spectral_profile=scenario.spectral_profile,
        model_size=model_size,
        ppl_value=ppl_value,
        ppl_unc=ppl_unc,
        ppl_base_value=ppl_base_value,
        ppl_base_unc=ppl_base_unc,
        ppl_ratio_value=ppl_ratio_value,
        ppl_ratio_unc=ppl_ratio_unc,
        ppl_diff_value=ppl_diff_value,
        ppl_diff_unc=ppl_diff_unc,
        mean_kld_value=mean_kld_value,
        mean_kld_unc=mean_kld_unc,
    )


def run_perplexity(
    args: argparse.Namespace,
    scenario: Scenario,
    kl_base_file: str,
    use_kl_divergence: bool,
) -> Result:
    cmd = build_command(args, scenario, kl_base_file, use_kl_divergence)
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        sys.stderr.write(output)
        raise RuntimeError(f"llama-perplexity failed for {scenario.name}")

    try:
        return parse_result(output, scenario, Path(scenario.model).stat().st_size, use_kl_divergence)
    except RuntimeError as exc:
        snippet = "\n".join(output.strip().splitlines()[-80:])
        raise RuntimeError(f"{exc}\n--- llama-perplexity tail ---\n{snippet}") from exc


def annotate_deltas(results: list[Result]) -> None:
    baselines: dict[str, Result] = {}
    for result in results:
        if result.cache_mode == "legacy":
            baselines[result.weights_mode] = result

    for result in results:
        baseline = baselines.get(result.weights_mode)
        if baseline is None or baseline.ppl_value == 0:
            continue
        result.ppl_delta_vs_legacy_pct = 100.0 * (result.ppl_value / baseline.ppl_value - 1.0)


def validate_results(args: argparse.Namespace, results: list[Result]) -> None:
    if not args.require_kld_metrics:
        return

    for result in results:
        if result.cache_mode == "legacy":
            continue
        if result.mean_kld_value is None or result.ppl_ratio_value is None or result.ppl_diff_value is None:
            raise RuntimeError(
                f"missing KL-divergence metrics for {result.scenario}: "
                f"mean_kld={result.mean_kld_value} ppl_ratio={result.ppl_ratio_value} ppl_diff={result.ppl_diff_value}"
            )


def format_metric(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def format_size_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.3f}"


def print_summary(results: Iterable[Result]) -> None:
    print("| 🧪 Сценарий | Весы | KV | Профиль | PPL | ΔPPL vs fp base | PPL ratio | Mean KLD | ΔPPL vs legacy same weights | Model GiB |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            "| {scenario} | {weights_mode} | {cache_mode} | {profile} | {ppl} | {ppl_diff} | {ppl_ratio} | {kld} | {legacy_delta} | {size} |".format(
                scenario=result.scenario,
                weights_mode=result.weights_mode,
                cache_mode=result.cache_mode,
                profile=result.spectral_profile or "-",
                ppl=format_metric(result.ppl_value),
                ppl_diff=format_metric(result.ppl_diff_value),
                ppl_ratio=format_metric(result.ppl_ratio_value),
                kld=format_metric(result.mean_kld_value, digits=6),
                legacy_delta=format_delta(result.ppl_delta_vs_legacy_pct),
                size=format_size_gib(result.model_size),
            )
        )


def main() -> int:
    args = parse_args()
    scenarios = build_scenarios(args)

    baseline_scenario = scenarios[0]

    if args.kl_base_file:
        kl_base_file = str(Path(args.kl_base_file))
        Path(kl_base_file).parent.mkdir(parents=True, exist_ok=True)
        baseline_result = run_perplexity(args, baseline_scenario, kl_base_file, use_kl_divergence=False)
        results = [baseline_result]
    else:
        with tempfile.TemporaryDirectory(prefix="spectralquant-ppl-") as tmp_dir:
            kl_base_file = str(Path(tmp_dir) / "baseline.kld")
            baseline_result = run_perplexity(args, baseline_scenario, kl_base_file, use_kl_divergence=False)
            results = [baseline_result]

            for scenario in scenarios[1:]:
                results.append(run_perplexity(args, scenario, kl_base_file, use_kl_divergence=True))

            annotate_deltas(results)
            validate_results(args, results)
            print_summary(results)

            if args.output_json:
                output_path = Path(args.output_json)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "perplexity_bin": str(Path(args.perplexity_bin)),
                    "legacy_model": args.legacy_model,
                    "spectral_model": args.spectral_model,
                    "spectral_calibration": args.spectral_calibration,
                    "dataset": args.dataset,
                    "cache_type": args.cache_type,
                    "ctx_size": args.ctx_size,
                    "batch": args.batch,
                    "ubatch": args.ubatch,
                    "chunks": args.chunks,
                    "ppl_stride": args.ppl_stride,
                    "threads": args.threads,
                    "n_gpu_layers": args.n_gpu_layers,
                    "results": [asdict(result) for result in results],
                }
                output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            return 0

    for scenario in scenarios[1:]:
        results.append(run_perplexity(args, scenario, kl_base_file, use_kl_divergence=True))

    annotate_deltas(results)
    validate_results(args, results)
    print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "perplexity_bin": str(Path(args.perplexity_bin)),
            "legacy_model": args.legacy_model,
            "spectral_model": args.spectral_model,
            "spectral_calibration": args.spectral_calibration,
            "dataset": args.dataset,
            "cache_type": args.cache_type,
            "ctx_size": args.ctx_size,
            "batch": args.batch,
            "ubatch": args.ubatch,
            "chunks": args.chunks,
            "ppl_stride": args.ppl_stride,
            "threads": args.threads,
            "n_gpu_layers": args.n_gpu_layers,
            "kl_base_file": kl_base_file,
            "results": [asdict(result) for result in results],
        }
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
