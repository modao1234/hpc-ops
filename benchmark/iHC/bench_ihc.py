#!/usr/bin/env python3
# Copyright (C) 2026 Tencent.

"""iHC operator benchmark — CUDA-graph event timing, baseline save/check.

Usage:
    # record baseline from the current build
    python3 benchmark/iHC/bench_ihc.py --jsonl baseline.jsonl

    # run and compare against a saved baseline (exit 1 on >5% regression)
    python3 benchmark/iHC/bench_ihc.py --check baseline.jsonl

    # quick smoke test
    python3 benchmark/iHC/bench_ihc.py --ops pre post_pre --hidden 6144 --warmup 2 --iters 10

Timing: CUDA Graph capture → replay under event pair → median over --iters samples.
Each sample is one graph replay (one operator call at the given shape).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Optional

import torch

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))
import hpc  # noqa: E402

# ---------------------------------------------------------------------------
# Dispatch-region thresholds (mirror entry.cc / iHC.cu to cover every branch)
# ---------------------------------------------------------------------------
# post_pre non-SM90: cluster < 80 (6144) or <= 128 (3072/4096),
#                    retained <= 296, rows-1 < 297, rows-2 >= 297
# pre splitk threshold is device-dependent; cover small and large batch.
BATCH_SIZES = [1, 4, 8, 16, 32, 64, 80, 128, 256, 297, 320, 512, 1024, 2048, 4096, 8192, 16384]
HIDDEN_DIMS = [3072, 4096, 6144]
HC_MULT = 4
NORM_EPS = 1e-5
HC_EPS = 1e-6
MAGNITUDE = 2.0
RMS_EPS = 1e-5


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def bench_us(fn: Callable, warmup: int, iters: int) -> float:
    """Return median latency in microseconds using CUDA Graph + event timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    torch.cuda.synchronize()

    # warmup replays
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()

    samples = [s.elapsed_time(e) * 1e3 for s, e in pairs]  # ms → µs
    return median(samples)


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def _make_pre_inputs(num_batch: int, hidden_dim: int, fuse_rms_norm: bool):
    dev = "cuda"
    x = torch.rand((num_batch, HC_MULT, hidden_dim), dtype=torch.bfloat16, device=dev)
    w = torch.rand((2 * HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((2,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((2 * HC_MULT,), dtype=torch.float32, device=dev)
    rms_w = (
        torch.rand((hidden_dim,), dtype=torch.bfloat16, device=dev) if fuse_rms_norm else None
    )
    if fuse_rms_norm:
        return lambda: hpc.fuse_ihc_pre(
            x, w, hc_scale, hc_base, NORM_EPS, HC_EPS, MAGNITUDE, rms_w, RMS_EPS
        )
    return lambda: hpc.fuse_ihc_pre(x, w, hc_scale, hc_base, NORM_EPS, HC_EPS, MAGNITUDE)


def _make_head_inputs(num_batch: int, hidden_dim: int, fuse_rms_norm: bool):
    dev = "cuda"
    x = torch.rand((num_batch, HC_MULT, hidden_dim), dtype=torch.bfloat16, device=dev)
    w = torch.rand((HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((1,), dtype=torch.float32, device=dev)      # head: last dim must be 1
    hc_base = torch.rand((HC_MULT,), dtype=torch.float32, device=dev)
    rms_w = (
        torch.rand((hidden_dim,), dtype=torch.bfloat16, device=dev) if fuse_rms_norm else None
    )
    if fuse_rms_norm:
        # hpc.fuse_ihc_head does not expose rms_weight; go through the registered op.
        return lambda: torch.ops.hpc.fuse_ihc_head(
            x, w, hc_scale, hc_base, NORM_EPS, HC_EPS, rms_w, RMS_EPS
        )
    return lambda: hpc.fuse_ihc_head(x, w, hc_scale, hc_base, NORM_EPS, HC_EPS)


def _make_post_inputs(num_batch: int, hidden_dim: int):
    dev = "cuda"
    x = torch.rand((num_batch, hidden_dim), dtype=torch.bfloat16, device=dev)
    residual = torch.rand((num_batch, HC_MULT, hidden_dim), dtype=torch.bfloat16, device=dev)
    H_post = torch.rand((num_batch, HC_MULT), dtype=torch.float32, device=dev)
    return lambda: hpc.fuse_ihc_post(x, residual, H_post)


def _make_post_pre_inputs(num_batch: int, hidden_dim: int, fuse_rms_norm: bool):
    dev = "cuda"
    xa = torch.rand((num_batch, hidden_dim), dtype=torch.bfloat16, device=dev)
    residual = torch.rand((num_batch, HC_MULT, hidden_dim), dtype=torch.bfloat16, device=dev)
    H_post_in = torch.rand((num_batch, HC_MULT), dtype=torch.float32, device=dev)
    w = torch.rand((2 * HC_MULT, HC_MULT * hidden_dim), dtype=torch.float32, device=dev) * 6e-3
    hc_scale = torch.rand((2,), dtype=torch.float32, device=dev)
    hc_base = torch.rand((2 * HC_MULT,), dtype=torch.float32, device=dev)
    rms_w = (
        torch.rand((hidden_dim,), dtype=torch.bfloat16, device=dev) if fuse_rms_norm else None
    )
    if fuse_rms_norm:
        return lambda: hpc.fuse_ihc_post_pre(
            xa, residual, H_post_in, w, hc_scale, hc_base,
            NORM_EPS, HC_EPS, MAGNITUDE, rms_w, RMS_EPS
        )
    return lambda: hpc.fuse_ihc_post_pre(
        xa, residual, H_post_in, w, hc_scale, hc_base,
        NORM_EPS, HC_EPS, MAGNITUDE
    )


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------
@dataclass
class BenchCase:
    op: str          # pre | head | post | post_pre
    num_batch: int
    hidden_dim: int
    fuse_rms_norm: bool


def all_cases(ops: list[str], hidden_dims: list[int], batch_sizes: list[int]) -> list[BenchCase]:
    cases = []
    for op in ops:
        for hidden_dim in hidden_dims:
            for num_batch in batch_sizes:
                if op in ("pre", "head", "post_pre"):
                    for norm in (False, True):
                        cases.append(BenchCase(op, num_batch, hidden_dim, norm))
                else:
                    cases.append(BenchCase(op, num_batch, hidden_dim, False))
    return cases


def make_fn(case: BenchCase) -> Callable:
    if case.op == "pre":
        return _make_pre_inputs(case.num_batch, case.hidden_dim, case.fuse_rms_norm)
    if case.op == "head":
        return _make_head_inputs(case.num_batch, case.hidden_dim, case.fuse_rms_norm)
    if case.op == "post":
        return _make_post_inputs(case.num_batch, case.hidden_dim)
    if case.op == "post_pre":
        return _make_post_pre_inputs(case.num_batch, case.hidden_dim, case.fuse_rms_norm)
    raise ValueError(f"unknown op: {case.op}")


def case_key(case: BenchCase) -> str:
    norm_tag = f"_norm" if case.fuse_rms_norm else ""
    return f"{case.op}/h{case.hidden_dim}/n{case.num_batch}{norm_tag}"


# ---------------------------------------------------------------------------
# Baseline save / compare
# ---------------------------------------------------------------------------
def save_jsonl(results: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"saved {len(results)} rows → {path}")


def load_jsonl(path: str) -> dict[str, float]:
    baseline: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            baseline[row["key"]] = row["median_us"]
    return baseline


def compare(results: list[dict], baseline: dict[str, float], threshold: float) -> bool:
    """Print regression table. Returns True only if at least one case was compared
    and no case exceeded the threshold. A missing baseline entry is an error, not a pass."""
    regressions = []
    missing = []
    compared = 0
    for row in results:
        key = row["key"]
        if key not in baseline:
            missing.append(key)
            continue
        base_us = baseline[key]
        head_us = row["median_us"]
        if base_us is None or head_us is None:
            print(f"  {'SKIP':9s}  {key:50s}  (OOM or error on this side)")
            continue
        compared += 1
        ratio = head_us / base_us if base_us > 0 else float("inf")
        status = "OK" if ratio <= 1.0 + threshold else "REGRESSED"
        if status == "REGRESSED":
            regressions.append((key, base_us, head_us, ratio))
        marker = " !" if status == "REGRESSED" else ""
        print(f"  {status:9s}  {key:50s}  base={base_us:8.2f}us  head={head_us:8.2f}us  "
              f"ratio={ratio:.3f}{marker}")

    if missing:
        print(f"\n{len(missing)} case(s) have no baseline entry - re-record the baseline "
              f"with a matching --ops/--hidden/--batch sweep:")
        for key in missing[:10]:
            print(f"  MISSING  {key}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if compared == 0:
        print("\nNOTHING COMPARED. This is not a pass.")
        return False

    if regressions:
        print(f"\n{len(regressions)} of {compared} compared case(s) exceed "
              f"{threshold*100:.0f}% threshold:")
        for key, base_us, head_us, ratio in regressions:
            print(f"  {key}  {base_us:.2f} -> {head_us:.2f} us  ({ratio:.3f}x)")
        return False
    print(f"\n{compared} case(s) compared, all within {threshold*100:.0f}% threshold.")
    return not missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_OPS = ["pre", "head", "post", "post_pre"]


def busy_fraction(gpu: int) -> Optional[int]:
    """Return the current GPU utilization percent, or None if it cannot be read."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        return int(out.stdout.strip().split()[0])
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ops", nargs="+", default=ALL_OPS, choices=ALL_OPS,
                   help="operators to benchmark (default: all)")
    p.add_argument("--hidden", nargs="+", type=int, default=HIDDEN_DIMS,
                   help="hidden_dim values to sweep")
    p.add_argument("--batch", nargs="+", type=int, default=BATCH_SIZES,
                   help="num_batch values to sweep")
    p.add_argument("--warmup", type=int, default=5, help="graph replay warmup iterations")
    p.add_argument("--iters", type=int, default=100, help="timed iterations per case")
    p.add_argument("--jsonl", type=str, default=None,
                   help="write results to this JSONL file (baseline save)")
    p.add_argument("--check", type=str, default=None,
                   help="compare against this baseline JSONL; exit 1 on regression")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="regression threshold as a fraction (default 0.05 = 5%%)")
    p.add_argument("--gpu", type=int, default=4,
                   help="CUDA device index (default 4; pick an idle card)")
    p.add_argument("--max-util", type=int, default=10,
                   help="refuse to record a baseline on a card above this utilization %%")
    p.add_argument("--allow-busy", action="store_true",
                   help="override the utilization guard")
    return p.parse_args()


def main():
    args = parse_args()
    util = busy_fraction(args.gpu)
    torch.cuda.set_device(args.gpu)

    sm = torch.cuda.get_device_capability(args.gpu)
    arch = sm[0] * 10 + sm[1]
    device_name = torch.cuda.get_device_name(args.gpu)
    print(f"device: {device_name}  sm{arch}  gpu={args.gpu}  util={util}%")
    if util is None:
        print("WARNING: could not read GPU utilization; results may be noisy.")
    elif util > args.max_util and not args.allow_busy:
        sys.exit(
            f"REFUSING: gpu {args.gpu} is at {util}% utilization (limit {args.max_util}%).\n"
            f"Timings on a busy card are not comparable. Pick an idle card, or pass "
            f"--allow-busy if you only need a rough number.")
    elif util > args.max_util:
        print(f"WARNING: gpu {args.gpu} at {util}% utilization; timings will be unreliable.")
    print(f"hpc loaded arch: {hpc.__loaded_arch__}")
    print(f"ops={args.ops}  hidden={args.hidden}  batch={args.batch}")
    print(f"warmup={args.warmup}  iters={args.iters}  threshold={args.threshold*100:.0f}%")
    print()

    cases = all_cases(args.ops, args.hidden, args.batch)
    results = []
    for i, case in enumerate(cases):
        key = case_key(case)
        try:
            fn = make_fn(case)
            us = bench_us(fn, args.warmup, args.iters)
            results.append({"key": key, "op": case.op, "num_batch": case.num_batch,
                            "hidden_dim": case.hidden_dim, "fuse_rms_norm": case.fuse_rms_norm,
                            "median_us": us, "arch": arch, "device": device_name})
            print(f"[{i+1:4d}/{len(cases)}]  {key:50s}  {us:8.2f} µs")
        except Exception as e:
            print(f"[{i+1:4d}/{len(cases)}]  {key:50s}  ERROR: {e}")
            results.append({"key": key, "op": case.op, "num_batch": case.num_batch,
                            "hidden_dim": case.hidden_dim, "fuse_rms_norm": case.fuse_rms_norm,
                            "median_us": None, "error": str(e), "arch": arch,
                            "device": device_name})

    print()
    if args.jsonl:
        save_jsonl(results, args.jsonl)

    if args.check:
        baseline = load_jsonl(args.check)
        print(f"\nComparing against baseline: {args.check}  ({len(baseline)} entries)\n")
        ok = compare(results, baseline, args.threshold)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
