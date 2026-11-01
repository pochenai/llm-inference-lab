"""Compare matched Rust Runner/SGLang Engine cold/warm cohorts, without HTTP.

Run from the repository root with the installed SGLang environment:
  ~/.venvs/py3.12/bin/python rust/engine/benches/qwen2_prefix_cache_compare.py \
    --model-dir MODEL_DIR --include-graph-control

BF16/overlap: 4K inputs, prefixes 0/2048/3584, batches 1/2/4, one output;
optional Graph control: prefix 2048, 64 outputs, batches 1/2/4. Defaults are two
warmups and three samples per state in three independent engine processes.
Rotate engine order; retain frozen sources/binary, exact fixtures, commands and
raw samples. Medians of per-process means are distinct from saturated throughput.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
from statistics import mean, median
import subprocess
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[0, 2048, 3584])
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--num-slots", type=int, default=20480)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--include-graph-control", action="store_true")
    args = parser.parse_args()
    assert min(*args.batch_size, args.tokens, args.num_slots, args.warmup, args.iterations, args.repetitions) > 0
    assert all(0 <= n < args.tokens for n in args.prefix_tokens)
    assert not args.include_graph_control or args.tokens > 2048
    out = (args.output_dir or REPO / "rust/profiling/qwen2" / datetime.now(timezone.utc).strftime("prefix-cache-comparison-%Y%m%d-%H%M%S")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    overrides = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_HOME": "/usr/local/cuda", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MAX_JOBS": "2", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    env = dict(os.environ, **overrides)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", os.defpath)
    commands: list[dict[str, Any]] = []

    def run(label: str, command: list[str]) -> None:
        commands.append({"label": label, "argv": command})
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(f"RUN {label}", flush=True)
        start = time.perf_counter()
        with (out / f"{label}.jsonl").open("w") as stdout, (out / f"{label}.stderr").open("w") as stderr:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = child.wait(timeout=1800)
            except BaseException:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
                raise
        assert code == 0, f"{label} exited {code}; inspect {out / (label + '.stderr')}"
        print(f"DONE {label} ({time.perf_counter() - start:.1f}s)", flush=True)

    run("build", ["cargo", "bench", "--manifest-path", "rust/Cargo.toml", "--offline", "--locked", "-j", "1", "-p", "engine", "--bench", "qwen2_prefix_cache", "--no-run", "--message-format=json"])
    artifacts = [json.loads(line) for line in (out / "build.jsonl").read_text().splitlines()]
    built = next(row for row in artifacts if row.get("reason") == "compiler-artifact" and row["target"]["name"] == "qwen2_prefix_cache" and row.get("executable"))
    assert built["profile"]["opt_level"] != "0"
    binary = out / "rust-prefix-cache"
    shutil.copy2(built["executable"], binary)
    for name in ["qwen2_prefix_cache.rs", "sglang_qwen2_prefix_cache.py", "qwen2_prefix_cache_compare.py"]:
        shutil.copy2(REPO / "rust/engine/benches" / name, out / name)
    manifest = {
        "engine_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "environment_overrides": overrides,
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True).strip(),
        "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [binary, *[out / n for n in ["qwen2_prefix_cache.rs", "sglang_qwen2_prefix_cache.py", "qwen2_prefix_cache_compare.py"]]]},
        "protocol": vars(args) | {"model_dir": str(args.model_dir.resolve()), "output_dir": str(out)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    groups: dict[str, list[dict[str, Any]]] = {"eager-prefill": [], "graph-generation": []}
    for group, output, prefixes in [("eager-prefill", 1, args.prefix_tokens), ("graph-generation", 64, [2048] if args.include_graph_control else [])]:
        for batch in args.batch_size:
            for prefix in prefixes:
                suffixes: list[int] = []
                for token in range(1000, len(tokenizer)):
                    word = tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if not word.startswith(" ") or not all(c.isascii() and c.isalpha() for c in word.strip()):
                        continue
                    ids = [9707] * prefix + [token] * (args.tokens - prefix)
                    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if tokenizer.encode(text, add_special_tokens=True) == ids:
                        suffixes.append(token)
                    if len(suffixes) == batch:
                        break
                assert len(suffixes) == batch
                groups[group].append({"name": f"{group}-b{batch}-p{prefix}", "batch_size": batch, "tokens": args.tokens, "prefix_tokens": prefix, "output_tokens": output, "shared_token_id": 9707, "suffix_token_ids": suffixes})
    for group, cases in groups.items():
        (out / f"{group}-fixtures.json").write_text(json.dumps(cases, indent=2) + "\n")
    rows: list[dict[str, Any]] = []
    expected: dict[str, list[str]] = {}
    for repetition in range(args.repetitions):
        for group, cases in groups.items():
            if not cases:
                continue
            for engine in (["rust", "sglang"] if repetition % 2 == 0 else ["sglang", "rust"]):
                graph = ["--cuda-graph"] if group == "graph-generation" else []
                if engine == "sglang":
                    label = f"r{repetition}-{group}-sglang"
                    run(label, [sys.executable, str(out / "sglang_qwen2_prefix_cache.py"), "--model-dir", str(args.model_dir.resolve()), "--fixtures", str(out / f"{group}-fixtures.json"), "--num-slots", str(args.num_slots), "--warmup", str(args.warmup), "--iterations", str(args.iterations), *graph])
                    data = [json.loads(line) for line in (out / f"{label}.jsonl").read_text().splitlines()]
                    assert len([r for r in data if r["event"] == "sample"]) == len(cases) * 2 * (args.warmup + args.iterations)
                    for case in cases:
                        selected = [r for r in data if r.get("case") == case["name"] and not r["warmup"]]
                        for state in ["cold", "warm"]:
                            samples = [r["sample"] for r in selected if r["state"] == state]
                            rows.append(summarize(case, engine, repetition, state, samples, args.iterations, expected))
                else:
                    for case in cases:
                        label = f"r{repetition}-{case['name']}-rust"
                        run(label, [str(binary), "--model-dir", str(args.model_dir.resolve()), "--tokens", str(args.tokens), "--prefix-tokens", str(case["prefix_tokens"]), "--batch-size", str(case["batch_size"]), "--output-tokens", str(case["output_tokens"]), "--num-slots", str(args.num_slots), "--cache-tokens", str(args.num_slots), "--warmup", str(args.warmup), "--iterations", str(args.iterations), *graph])
                        data = [json.loads(line) for line in (out / f"{label}.jsonl").read_text().splitlines()]
                        config = data[0]
                        assert config["suffix_token_ids"] == case["suffix_token_ids"] and config["num_slots"] == args.num_slots
                        assert all(r["kv_released"] for r in data if r["event"] == "sample")
                        for state in ["cold", "warm"]:
                            samples = [r["sample"] for r in data if r["event"] == "sample" and not r["warmup"] and r["state"] == state]
                            rows.append(summarize(case, engine, repetition, state, samples, args.iterations, expected))
                (out / "process-means.json").write_text(json.dumps(rows, indent=2) + "\n")
    result = []
    for cases in groups.values():
        for case in cases:
            entry: dict[str, Any] = dict(case)
            for engine in ["rust", "sglang"]:
                for state in ["cold", "warm"]:
                    selected = [r for r in rows if r["case"] == case["name"] and r["engine"] == engine and r["state"] == state]
                    assert len(selected) == args.repetitions
                    entry[f"{engine}_{state}_ms"] = median(r["mean_seconds"] for r in selected) * 1000
            entry["rust_warm_latency_reduction_percent"] = (1 - entry["rust_warm_ms"] / entry["sglang_warm_ms"]) * 100
            entry["rust_output_text"] = expected["rust:" + case["name"]]
            entry["sglang_output_text"] = expected["sglang:" + case["name"]]
            entry["cross_engine_text_match"] = entry["rust_output_text"] == entry["sglang_output_text"]
            result.append(entry)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "summary": result}, indent=2), flush=True)


def summarize(case: dict[str, Any], engine: str, repetition: int, state: str, samples: list[dict[str, Any]], iterations: int, expected: dict[str, list[str]]) -> dict[str, Any]:
    assert len(samples) == iterations
    for sample in samples:
        assert sample["output_text"] == expected.setdefault(engine + ":" + case["name"], sample["output_text"]), f"within-engine text mismatch: {case['name']} {engine}"
        reused = sample["counters"]["reused_tokens"] if engine == "rust" else sample["reused_tokens"]
        assert reused == (case["batch_size"] * case["prefix_tokens"] if state == "warm" else 0)
        assert sample["computed_prefill_tokens"] == case["batch_size"] * case["tokens"] - reused
    first = [value for sample in samples for value in sample["time_to_first_text_seconds"] if value is not None]
    return {"case": case["name"], "engine": engine, "repetition": repetition, "state": state,
        "mean_seconds": mean(sample["cohort_wall_seconds"] for sample in samples),
        "first_text_mean_seconds": mean(first) if first else None,
        "raw_seconds": [sample["cohort_wall_seconds"] for sample in samples]}


if __name__ == "__main__":
    main()
