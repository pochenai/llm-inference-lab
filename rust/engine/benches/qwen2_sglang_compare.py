"""Compare BF16 static GPU-feedback steps, excluding HTTP, text decode and serving overlap.

From the repository root:
  /usr/bin/python3 rust/engine/benches/qwen2_sglang_compare.py --model-dir MODEL_DIR --include-sglang-graph

Defaults: B=1/2/4, input=32/4096, output=64, two warmups, three samples per
shape and three independent process repetitions. Rust is optimized/eager; SGLang
eager is the matched baseline, graph is an explicitly separate reference. Engines
run sequentially, rotating order between repetitions. No profiler is attached.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
from statistics import mean, median
import subprocess
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(samples: list[dict[str, float]]) -> dict[str, float]:
    """Preserve per-process aggregates; never pretend individual decode steps are trials."""
    return {key: median(sample[key] for sample in samples) for key in samples[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pattern", choices=["repeated", "ramp"], default="repeated")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sglang-python", type=Path, default=Path.home() / ".venvs/py3.12/bin/python")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 4096])
    parser.add_argument("--include-sglang-graph", action="store_true")
    args = parser.parse_args()
    if min(args.repetitions, args.warmup, args.iterations, args.output_tokens - 1, *args.batch_size, *args.tokens) < 1:
        parser.error("counts must be positive and output-tokens must exceed one")
    out = (args.results_dir or REPO / "rust/profiling/qwen2/sglang-comparison" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_HOME="/usr/local/cuda", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MAX_JOBS="2", TOKENIZERS_PARALLELISM="false", PYTORCH_ALLOC_CONF="expandable_segments:True")
    # FlashInfer invokes Ninja by name; use the selected Python environment's tools.
    env["PATH"] = str(args.sglang_python.parent.absolute()) + os.pathsep + env.get("PATH", os.defpath)
    commands: list[dict[str, Any]] = []

    def run(label: str, command: list[str]) -> None:
        commands.append({"label": label, "command": command})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(label, flush=True)
        with (out / f"{label}.stdout").open("w") as stdout, (out / f"{label}.stderr").open("w") as stderr:
            subprocess.run(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, check=True, timeout=1200)

    run("build", ["cargo", "bench", "--manifest-path", "rust/Cargo.toml", "--offline", "--locked", "-p", "engine", "--bench", "qwen2", "--no-run", "--message-format=json"])
    binary = next(Path(r["executable"]) for r in read_jsonl(out / "build.stdout") if r.get("reason") == "compiler-artifact" and r["target"]["name"] == "qwen2" and r.get("executable"))
    frozen = out / "qwen2"
    shutil.copy2(binary, frozen)
    for name in ["qwen2.rs", "sglang_qwen2.py", Path(__file__).name]:
        shutil.copy2(Path(__file__).with_name(name), out / name)
    metadata = {
        "sglang_tool_directory": str(args.sglang_python.parent.absolute()),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "binary_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        "model_config_sha256": hashlib.sha256((args.model_dir / "config.json").read_bytes()).hexdigest(),
        "model_snapshot": str(args.model_dir.resolve()),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,temperature.gpu,pstate", "--format=csv"], text=True),
        "versions": subprocess.check_output([str(args.sglang_python), "-c", "import importlib.metadata as m; print({k:m.version(k) for k in ['sglang','torch','transformers','flashinfer-python']})"], text=True),
        "environment_overrides": {k: env[k] for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_HOME", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS", "TOKENIZERS_PARALLELISM", "PYTORCH_ALLOC_CONF"]},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "source.diff").write_bytes(subprocess.check_output(["git", "diff", "--", "rust/engine", ":!*.env", ":!.env*"], cwd=REPO))
    kv_slots = ((max(args.batch_size) * (max(args.tokens) + args.output_tokens) + 127) // 128) * 128
    cases = list(itertools.product(args.batch_size, args.tokens))
    checksums = {n: sum((i+1)*(9707+(i%4096 if args.input_pattern=="ramp" else 0)) for i in range(n)) for n in args.tokens}
    engines = ["rust-eager", "sglang-eager"] + (["sglang-graph"] if args.include_sglang_graph else [])
    results: dict[tuple[int, int, str], list[dict[str, float]]] = {}
    for repetition in range(args.repetitions):
        order = engines[repetition % len(engines):] + engines[:repetition % len(engines)]
        for engine in order:
            if engine == "rust-eager":
                for batch, tokens in cases:
                    label = f"r{repetition}-{engine}-b{batch}-t{tokens}"
                    run(label, [str(frozen), "--model-dir", str(args.model_dir.resolve()), "--dtype", "bf16", "--input-pattern", args.input_pattern, "--gpu-feedback", "--kv-slots", str(kv_slots), "--batch-size", str(batch), "--tokens", str(tokens), "--decode-steps", str(args.output_tokens - 1), "--warmup", str(args.warmup), "--iterations", str(args.iterations)])
                    records = read_jsonl(out / f"{label}.stdout")
                    loaded = next(r for r in records if r["event"] == "loaded")
                    assert loaded["dtype"] == "bf16" and loaded["gpu_feedback"] and loaded["kv_slots"] == kv_slots
                    assert loaded["input_pattern"] == args.input_pattern and loaded["input_checksum"] == checksums[tokens]
                    measured = [r for r in records if r["event"] == "sample" and not r["warmup"]]
                    assert len(measured) == args.iterations
                    assert sum(r.get("host_validation", False) for r in records) == 1
                    for sample in measured:
                        assert sample["gpu_feedback"] and sample["forward_count"] == args.output_tokens
                        assert sample["kv_released"] and sample["workspace_pool"]["in_use_bytes"] == 0
                    values = {
                        "prefill_seconds": mean(r["prefill_seconds"] for r in measured),
                        "decode_step_median_seconds": median(median(r["decode_step_seconds"]) for r in measured),
                        "total_seconds": mean(r["prefill_seconds"] + r["decode_seconds"] for r in measured),
                        "pool_used_peak_bytes": max(r["cuda_pool_used_peak_bytes"] for r in measured),
                    }
                    results.setdefault((batch, tokens, engine), []).append(values)
            else:
                label = f"r{repetition}-{engine}"
                output = out / f"{label}.jsonl"
                command = [str(args.sglang_python), str(out / "sglang_qwen2.py"), "--model-path", str(args.model_dir.resolve()), "--dtype", "bfloat16", "--comparison-input-pattern", args.input_pattern, "--mem-fraction-static", "0.5", "--max-total-tokens", str(kv_slots), "--max-running-requests", str(max(args.batch_size)), "--context-length", "8192", "--batch-size", *map(str, args.batch_size), "--input-len", *map(str, args.tokens), "--output-len", str(args.output_tokens), "--comparison-warmup", str(args.warmup), "--comparison-iterations", str(args.iterations), "--result-filename", str(output)]
                if engine == "sglang-eager":
                    command.append("--disable-cuda-graph")
                run(label, command)
                records = read_jsonl(output)
                assert len(records) == len(cases), "SGLang skipped a case"
                assert {(r["batch_size"], r["input_len"]) for r in records} == set(cases)
                for record in records:
                    assert record["output_len"] == args.output_tokens and record["dtype"] == "bfloat16"
                    assert record["input_pattern"] == args.input_pattern and record["input_checksum"] == checksums[record["input_len"]]
                    assert record["cuda_graph"] == (engine == "sglang-graph")
                    samples = record["samples"]
                    assert len(samples) == args.iterations
                    results.setdefault((record["batch_size"], record["input_len"], engine), []).append({
                        "prefill_seconds": mean(s["prefill_latency"] for s in samples),
                        "decode_step_median_seconds": median(s["median_decode_latency"] for s in samples),
                        "total_seconds": mean(s["total_latency"] for s in samples),
                        "pool_used_peak_bytes": max(s["torch_allocated_peak_bytes"] for s in samples),
                    })
            # Save independent process aggregates after each successful engine run.
            (out / "comparison.json").write_text(json.dumps([{"batch_size": b, "input_len": t, "engine": e, "processes": p, "median": summarize(p)} for (b, t, e), p in results.items()], indent=2) + "\n")
    lines = ["# Qwen BF16 static-model comparison", "", "Three-way modes are separate: Rust eager, SGLang eager, optional SGLang decode CUDA Graph.", "", "Ratios are Rust latency / SGLang latency; >1 means Rust is slower. Each cell is the median of independent process aggregates, not a pooled per-token statistic.", "", "| Batch | Input/request | Engine | Prefill ms | Decode median ms/step | Total ms | Output tokens/s |", "| ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
    for batch, tokens in cases:
        for engine in engines:
            value = summarize(results[(batch, tokens, engine)])
            lines.append(f"| {batch} | {tokens} | {engine} | {value['prefill_seconds']*1000:.2f} | {value['decode_step_median_seconds']*1000:.3f} | {value['total_seconds']*1000:.2f} | {batch*args.output_tokens/value['total_seconds']:.1f} |")
    lines += ["", "## Eager ratios and independent-process variation", "", "Ratios below are Rust / SGLang eager latency; paired decode ranges cover the three process repetitions and are not confidence intervals.", "", "| Batch | Input/request | Prefill ratio | Decode ratio | Total ratio | Paired decode ratio min..max |", "| ---: | ---: | ---: | ---: | ---: | --- |"]
    for batch, tokens in cases:
        rust = summarize(results[(batch, tokens, "rust-eager")])
        sglang = summarize(results[(batch, tokens, "sglang-eager")])
        paired = [r["decode_step_median_seconds"] / s["decode_step_median_seconds"] for r, s in zip(results[(batch, tokens, "rust-eager")], results[(batch, tokens, "sglang-eager")])]
        lines.append(f"| {batch} | {tokens} | {rust['prefill_seconds']/sglang['prefill_seconds']:.3f} | {rust['decode_step_median_seconds']/sglang['decode_step_median_seconds']:.3f} | {rust['total_seconds']/sglang['total_seconds']:.3f} | {min(paired):.3f}..{max(paired):.3f} |")
    lines += ["", "## Observed allocator peaks", "", "Maxima across measured samples and process repetitions, MiB. Rust CUDA-pool used and SGLang torch allocated have different scopes; this is not whole-device VRAM or an allocator-efficiency ranking.", "", "| Batch | Input/request | Rust pool used | SGLang eager allocated | SGLang Graph allocated |", "| ---: | ---: | ---: | ---: | ---: |"]
    for batch, tokens in cases:
        peaks = [max(p["pool_used_peak_bytes"] for p in results[(batch, tokens, e)]) / 2**20 for e in engines]
        lines.append(f"| {batch} | {tokens} | " + " | ".join(f"{peak:.2f}" for peak in peaks) + (" |" if args.include_sglang_graph else " | n/a |"))
    lines += ["", f"Input pattern is {args.input_pattern}: repeated token 9707, or ramp 9707+(position%4096). KV token capacity is {kv_slots} in both engines; no prefix reuse, fixed {args.output_tokens} outputs, greedy and no EOS early stopping. Rust --gpu-feedback validates final IDs against an untimed host-feedback control, reads final IDs outside timers and does not retain all previous workspaces. SGLang uses its official synchronous static loop, not its serving overlap scheduler.", "", "Both timers include CPU preparation/submission and GPU completion. Rust input-view construction is outside timers; final D2H and text decode are excluded. SGLang has untimed device synchronization before each step; Rust forward awaits completion. Logit dtype/attention reduction rounding and kernel implementations are not identical even with BF16 weights/activations/KV. Graph reference accelerates decode only; Rust has no Qwen Graph yet.", "", "CUDA-pool used peaks (Rust) and torch allocated peaks (SGLang) are recorded with their allocator-specific meanings. They exclude whole-device/display memory; mapping/graph capacities differ, so they are not an allocator-efficiency ranking.", "", "Raw samples, backend class, graph capture logs, process aggregates, exact commands, frozen sources/binary and version/hardware metadata accompany this report. No Nsight profiler is attached to native samples. Results do not establish serving throughput, overlap parity, or output quality.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(out, flush=True)



if __name__ == "__main__":
    main()
