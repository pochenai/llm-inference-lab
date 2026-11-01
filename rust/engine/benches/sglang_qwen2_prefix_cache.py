"""SGLang 0.5.9 Engine-level streamed prefix-cache cohorts, without HTTP.

Run with ~/.venvs/py3.12/bin/python from the repository root:
  CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1 ~/.venvs/py3.12/bin/python \
    rust/engine/benches/sglang_qwen2_prefix_cache.py --model-dir MODEL_DIR \
    --fixtures FIXTURES_JSON --warmup 2 --iterations 3 --num-slots 20480

Fixtures contain the Rust benchmark's exact token patterns. Text encoding,
Scheduler, streaming and detokenization stay enabled. Flush KV between samples,
seed only the shared prefix for warm samples, retain model/JIT/Graph state.
Batch-tokenizer ingress sends one native batch packet before scheduling; no
SGLang model, cache or scheduler implementation is patched. Library logs go to
stderr and benchmark JSONL goes to stdout. Setup/flush/seeding are not timed.
For Nsight capture-range=cudaProfilerApi, pass --profile --cache-state warm
--iterations 1 with a single-case fixture. Native Engine.start_profile activities
CUDA_PROFILER starts capture on the GPU worker after the prefix seed; no PyTorch
profiler is layered over Nsight. Graph/node trace timings are diagnostic only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Iterator


@contextmanager
def library_output() -> Iterator[None]:
    # Native C++/child-process logging bypasses redirect_stdout; preserve JSONL fd 1.
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def cohort(engine: Any, prompts: list[str], output_tokens: int) -> dict[str, Any]:
    outputs: list[str] = [""] * len(prompts)
    first: list[float | None] = [None] * len(prompts)
    elapsed: list[float | None] = [None] * len(prompts)
    meta: list[dict[str, Any]] = [{} for _ in prompts]
    sampling = {"temperature": 0, "max_new_tokens": output_tokens, "ignore_eos": True}
    start = time.perf_counter()
    rids = [uuid.uuid4().hex for _ in prompts]
    with redirect_stdout(sys.stderr):
        for chunk in engine.generate(prompt=prompts, sampling_params=sampling, stream=True, rid=rids):
            index = chunk.get("index", 0)
            assert 0 <= index < len(prompts)
            seconds = time.perf_counter() - start
            assert chunk["meta_info"]["id"] == rids[index], (rids, chunk)
            text = chunk["text"]  # Native Engine streams cumulative text, not deltas.
            if text and first[index] is None:
                first[index] = seconds
            outputs[index] = text
            meta[index] = chunk["meta_info"]
            if meta[index].get("finish_reason") is not None:
                elapsed[index] = seconds
    wall = time.perf_counter() - start
    assert all(value is not None for value in elapsed), "missing request completion"
    assert all(value["completion_tokens"] == output_tokens and value["finish_reason"]["type"] == "length" for value in meta), meta
    return {
        "cohort_wall_seconds": wall,
        "time_to_first_text_seconds": first,
        "request_elapsed_seconds": elapsed,
        "output_text": outputs,
        "cached_tokens": [value["cached_tokens"] for value in meta],
        "prompt_tokens": [value["prompt_tokens"] for value in meta],
        "completion_tokens": [value["completion_tokens"] for value in meta],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--num-slots", type=int, default=20480)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cache-state", choices=["cold", "warm", "both"], default="both")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert min(args.num_slots, args.warmup, args.iterations) > 0
    version = importlib.metadata.version("sglang")
    assert version == "0.5.9", f"Recheck adapter for SGLang {version}"
    fixtures: list[dict[str, Any]] = json.loads(args.fixtures.read_text())
    assert fixtures
    assert not args.profile or (len(fixtures) == 1 and args.cache_state != "both" and args.iterations == 1), "profile one measured case/state"
    with library_output():
        import sglang
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        max_batch = max(case["batch_size"] for case in fixtures)
        context = max(8192, 8 + max(case["tokens"] + case["output_tokens"] for case in fixtures))
        engine = sglang.Engine(
            model_path=str(args.model_dir), dtype="bfloat16", random_seed=0,
            mem_fraction_static=0.5, max_total_tokens=args.num_slots,
            context_length=context, max_running_requests=max_batch,
            chunked_prefill_size=-1, max_prefill_tokens=max_batch * max(case["tokens"] for case in fixtures),
            disable_cuda_graph=not args.cuda_graph,
            cuda_graph_bs=sorted({1, *[case["batch_size"] for case in fixtures]}),
            cuda_graph_max_bs=max_batch, disable_overlap_schedule=False,
            page_size=1, radix_eviction_policy="lru", schedule_policy="fcfs",
            attention_backend="flashinfer", stream_interval=1,
            enable_tokenizer_batch_encode=True, log_level="warning",
        )
    try:
        # generate starts TokenizerManager's response loop; control RPCs need it first.
        cohort(engine, ["Hello"], 1)
        with library_output():
            info = engine.get_server_info()
        print(json.dumps({"event": "configuration", "version": version, "server_info": info}), flush=True)
        assert info["max_total_num_tokens"] == args.num_slots, "actual KV capacity differs"
        for case in fixtures:
            prefix = [case["shared_token_id"]] * case["prefix_tokens"]
            ids = [prefix + [suffix] * (case["tokens"] - len(prefix)) for suffix in case["suffix_token_ids"]]
            texts = [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in ids]
            assert all(tokenizer.encode(text, add_special_tokens=True) == tokens for text, tokens in zip(texts, ids))
            assert engine.tokenizer_manager.tokenizer(texts)["input_ids"] == ids, "engine tokenizer differs"
            shared = tokenizer.decode(prefix, skip_special_tokens=False)
            assert tokenizer.encode(shared, add_special_tokens=True) == prefix
            expected: list[str] | None = None
            for iteration in range(args.warmup + args.iterations):
                states = (["cold", "warm"] if iteration % 2 == 0 else ["warm", "cold"]) if args.cache_state == "both" else [args.cache_state]
                for state in states:
                    with library_output():
                        flushed = engine.flush_cache()
                    assert flushed.success, "KV flush failed"
                    if state == "warm" and prefix:
                        seeded = cohort(engine, [shared], 1)
                        assert seeded["cached_tokens"] == [0]
                    profiling = args.profile and iteration >= args.warmup
                    if profiling:
                        with library_output():
                            engine.start_profile(activities=["CUDA_PROFILER"], with_stack=False, record_shapes=False)
                        import torch
                        profile_range = torch.cuda.nvtx.range_start("prefix_cache." + state)
                    try:
                        sample = cohort(engine, texts, case["output_tokens"])
                    finally:
                        if profiling:
                            torch.cuda.nvtx.range_end(profile_range)
                            with library_output():
                                engine.stop_profile()
                    cached = len(prefix) if state == "warm" else 0
                    assert sample["cached_tokens"] == [cached] * len(texts), sample
                    assert sample["prompt_tokens"] == [case["tokens"]] * len(texts), sample
                    assert expected is None or sample["output_text"] == expected, "cold/warm text differs"
                    expected = sample["output_text"]
                    sample["reused_tokens"] = cached * len(texts)
                    sample["computed_prefill_tokens"] = (case["tokens"] - cached) * len(texts)
                    print(json.dumps({"event": "sample", "case": case["name"], "state": state,
                        "iteration": iteration, "warmup": iteration < args.warmup, "sample": sample}), flush=True)
    finally:
        with library_output():
            engine.shutdown()


if __name__ == "__main__":
    main()
