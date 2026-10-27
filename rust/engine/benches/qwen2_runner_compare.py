"""Run alternating, independent Runner serial/overlap processes and Nsight captures.

From the repository root:
  /usr/bin/python3 rust/engine/benches/qwen2_runner_compare.py --model-dir MODEL_DIR --profile

Defaults: FP32/eager, batch 1/2/4, input 32/4096, 64 outputs, two warmups,
three measured cohorts per process and three independent processes per mode/case.
Each experiment gets a new directory; raw JSONL, commands, binary and traces remain
under rust/profiling/qwen2/runner-overlap. Profiler timings never determine speedup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from statistics import median
import subprocess
from typing import Any

REPO = Path(__file__).resolve().parents[3]
Interval = tuple[int, int]


def merged(intervals: list[Interval]) -> list[Interval]:
    result: list[Interval] = []
    for start, end in sorted(intervals):
        assert end >= start
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def duration(intervals: list[Interval]) -> int:
    return sum(end - start for start, end in merged(intervals))


def intersection(left: list[Interval], right: list[Interval]) -> int:
    a, b = merged(left), merged(right)
    total = i = j = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze_trace(path: Path, mode: str, steps: int, batch_size: int) -> dict[str, Any]:
    """Correlate actual submissions, token D2H and synchronous CPU token acceptance."""
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        ranges = db.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start").fetchall()
        cohorts = [row for row in ranges if row["text"] == "runner_bench.sample"]
        assert len(cohorts) == 1, "Profile exactly one measured cohort"
        first, last = int(cohorts[0]["start"]), int(cohorts[0]["end"])
        ranges = [row for row in ranges if first <= row["start"] < row["end"] <= last]
        wrappers = [row for row in ranges if row["text"] == f"runner_bench.{mode}." + ("forward" if mode == "serial" else "enqueue")]
        submits = [row for row in ranges if row["text"] in {"qwen2.prefill.submit", "qwen2.decode.submit"}]
        accepts = [(int(row["start"]), int(row["end"])) for row in ranges if row["text"] == "runner.accept_token"]
        assert len(wrappers) == len(submits) == steps, (len(wrappers), len(submits), steps)
        assert len(accepts) == steps * batch_size, len(accepts)
        kernels = db.execute(
            "SELECT k.start,k.end,k.streamId,a.start AS api_start,a.end AS api_end,a.globalTid "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME a "
            "ON a.correlationId=k.correlationId WHERE k.start>=? AND k.end<=? ORDER BY k.start",
            (first, last),
        ).fetchall()
        copies = db.execute("SELECT start,end,copyKind,bytes,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? ORDER BY start", (first, last)).fetchall()
        d2h = [row for row in copies if row["copyKind"] == 2]
        assert len(d2h) == steps and all(row["bytes"] == batch_size * 4 for row in d2h), "Expected one sampled-ID D2H per batch forward"
        gpu_forwards: list[Interval] = []
        for scope in submits:
            selected = [row for row in kernels if row["globalTid"] == scope["globalTid"] and scope["start"] <= row["api_start"] < row["api_end"] <= scope["end"]]
            assert selected
            gpu_forwards.append((min(row["start"] for row in selected), max(row["end"] for row in selected)))
        compute = [(int(row["start"]), int(row["end"])) for row in kernels]
        copy_intervals = [(int(row["start"]), int(row["end"])) for row in copies]
        d2h_intervals = [(int(row["start"]), int(row["end"])) for row in d2h]
        assert len({row["streamId"] for row in kernels}) == 1, "Expected one ordered compute stream"
        assert not {row["streamId"] for row in kernels}.intersection(row["streamId"] for row in d2h), "D2H must use an independent stream"
        activities = compute + copy_intervals
        span = max(end for _, end in activities) - min(start for start, _ in activities)
        # Verify complete submission, rather than merely entering the enqueue call.
        before_readback = sum(wrappers[index + 1]["end"] < d2h[index]["start"] for index in range(steps - 1))
        before_gpu_end = sum(wrappers[index + 1]["end"] < gpu_forwards[index][1] for index in range(steps - 1))
        start_before_gpu_end = sum(wrappers[index + 1]["start"] < gpu_forwards[index][1] for index in range(steps - 1))
        assert before_readback == (0 if mode == "serial" else steps - 1), "Submission/readback ordering does not match execution mode"
        return {
            "mode": mode, "forwards": steps, "requests": batch_size, "kernels": len(kernels),
            "compute_stream_ids": sorted({row["streamId"] for row in kernels}),
            "d2h_stream_ids": sorted({row["streamId"] for row in d2h}),
            "next_submission_before_previous_d2h": before_readback,
            "next_submission_before_previous_gpu_end": before_gpu_end,
            "next_submission_start_before_previous_gpu_end": start_before_gpu_end,
            "adjacent_forward_pairs": steps - 1,
            "gpu_activity_span_ms": span / 1e6,
            "gpu_busy_union_ms": duration(activities) / 1e6,
            "gpu_gap_ms": (span - duration(activities)) / 1e6,
            "prefill_gpu_kernel_span_ms": (gpu_forwards[0][1] - gpu_forwards[0][0]) / 1e6,
            "decode_gpu_kernel_span_mean_ms": sum(end - start for start, end in gpu_forwards[1:]) / max(1, steps - 1) / 1e6,
            "cpu_accept_ms": duration(accepts) / 1e6,
            "cpu_accept_overlapping_compute_ms": intersection(accepts, compute) / 1e6,
            "h2d_bytes": sum(row["bytes"] for row in copies if row["copyKind"] == 1),
            "d2h_bytes": sum(row["bytes"] for row in d2h),
            "d2h_ms": duration(d2h_intervals) / 1e6,
            "d2h_overlapping_compute_ms": intersection(d2h_intervals, compute) / 1e6,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 4096])
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(*args.batch_size, *args.tokens, args.output_tokens, args.warmup, args.iterations, args.repetitions) < 1:
        parser.error("sizes/counts must be positive")
    root = args.results_dir or REPO / "rust/profiling/qwen2/runner-overlap" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    commands: list[list[str]] = []

    def run(command: list[str], label: str) -> None:
        commands.append(command)
        (root / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(label, flush=True)
        with (root / f"{label}.jsonl").open("w") as out, (root / f"{label}.stderr").open("w") as err:
            subprocess.run(command, cwd=REPO, stdout=out, stderr=err, check=True)

    run(["cargo", "bench", "--manifest-path", "rust/Cargo.toml", "--offline", "--locked", "-p", "engine", "--bench", "qwen2_runner", "--no-run", "--message-format=json"], "build")
    records = [json.loads(line) for line in (root / "build.jsonl").read_text().splitlines()]
    binary = next(Path(row["executable"]) for row in records if row.get("reason") == "compiler-artifact" and row["target"]["name"] == "qwen2_runner" and row.get("executable"))
    frozen_binary = root / "qwen2_runner"
    shutil.copy2(binary, frozen_binary)
    shutil.copy2(__file__, root / "qwen2_runner_compare.py")
    (root / "qwen2_runner.rs").write_text(Path(__file__).with_name("qwen2_runner.rs").read_text())
    (root / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--", "rust/engine", ":!*.env", ":!.env*"], cwd=REPO))
    metadata = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,pstate", "--format=csv"], text=True),
        "nsys": subprocess.check_output(["nsys", "--version"], text=True) if args.profile else None,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    expected: dict[tuple[int, int], list[str]] = {}
    base = [str(frozen_binary), "--model-dir", str(args.model_dir.resolve()), "--output-tokens", str(args.output_tokens), "--warmup", str(args.warmup)]
    for repetition in range(args.repetitions):
        for tokens in args.tokens:
            for batch in args.batch_size:
                for mode in (["serial", "overlap"] if repetition % 2 == 0 else ["overlap", "serial"]):
                    label = f"r{repetition}-b{batch}-t{tokens}-{mode}"
                    run(base + ["--mode", mode, "--tokens", str(tokens), "--batch-size", str(batch), "--iterations", str(args.iterations)], label)
                    rows = [json.loads(line) for line in (root / f"{label}.jsonl").read_text().splitlines()]
                    measured = [row for row in rows if row["event"] == "sample" and not row["warmup"]]
                    assert len(measured) == args.iterations
                    for row in measured:
                        assert row["output_text"] == expected.setdefault((batch, tokens), row["output_text"]), "Serial/overlap output mismatch"
                        assert row["generated_tokens"] == [args.output_tokens] * batch
                        assert row["forward_count"] == args.output_tokens and row["kv_released"]
                    summary = next(row for row in rows if row["event"] == "summary")
                    summary["cuda_pool_used_peak_bytes"] = max(row["cuda_pool_used_peak_bytes"] for row in measured)
                    summary["cuda_pool_reserved_peak_bytes"] = max(row["cuda_pool_reserved_peak_bytes"] for row in measured)
                    samples.setdefault((batch, tokens, mode), []).append(summary)
    comparisons: list[dict[str, Any]] = []
    for tokens in args.tokens:
        for batch in args.batch_size:
            serial = samples[(batch, tokens, "serial")]
            overlap = samples[(batch, tokens, "overlap")]
            comparisons.append({
                "batch_size": batch, "input_tokens": tokens, "output_tokens": args.output_tokens,
                "serial": serial, "overlap": overlap,
                "cohort_speedup_ratio_of_process_medians": median(row["cohort_mean_seconds"] for row in serial) / median(row["cohort_mean_seconds"] for row in overlap),
                "paired_process_speedup": [a["cohort_mean_seconds"] / b["cohort_mean_seconds"] for a, b in zip(serial, overlap)],
            })
    (root / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    traces: list[dict[str, Any]] = []
    if args.profile:
        for tokens in args.tokens:
            for batch in args.batch_size:
                for mode in ["serial", "overlap"]:
                    label = f"trace-b{batch}-t{tokens}-{mode}"
                    prefix = root / label
                    run(["nsys", "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none", "--event-sample=none", "--backtrace=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--output=" + str(prefix)] + base + ["--mode", mode, "--tokens", str(tokens), "--batch-size", str(batch), "--iterations", "1", "--profile"], label)
                    run(["nsys", "export", "--type=sqlite", "--output=" + str(prefix.with_suffix(".sqlite")), str(prefix.with_suffix(".nsys-rep"))], label + "-export")
                    traces.append({"batch_size": batch, "input_tokens": tokens, **analyze_trace(prefix.with_suffix(".sqlite"), mode, args.output_tokens, batch)})
                    (root / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps({"results_dir": str(root), "comparisons": len(comparisons), "traces": len(traces)}), flush=True)


def test_interval_union_and_intersection() -> None:
    assert duration([(0, 4), (2, 6), (8, 9)]) == 7
    assert intersection([(0, 4), (2, 6)], [(1, 3), (5, 8)]) == 3
    assert intersection([], [(1, 3)]) == 0
    assert intersection([(0, 1)], [(1, 3)]) == 0


if __name__ == "__main__":
    main()
