"""Adapt SGLang 0.5.9's official static-batch benchmark to our deterministic inputs.

From the repository root (MODEL_DIR is the local HF snapshot directory):
CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1 ~/.venvs/py3.12/bin/python rust/engine/benches/sglang_qwen2.py \
  --model-path MODEL_DIR --dtype bfloat16 --mem-fraction-static 0.5 \
  --context-length 8192 --disable-cuda-graph --batch-size 1 2 4 \
  --input-len 32 4096 --output-len 9 --comparison-warmup 2 --comparison-iterations 3 \
  --result-filename rust/profiling/qwen2/sglang-comparison/sglang-bf16-eager.jsonl

Omit --disable-cuda-graph for a separate graph-enabled baseline. Both engines use
token ID 9707, fresh KV per sample, one prefill and output_len - 1 decode forwards.
SGLang leaves sampled IDs on GPU; our Rust model includes a pinned CPU readback.
No SGLang model kernels are patched. This adapter targets the installed 0.5.9 API.
"""

import argparse
import importlib.metadata
import logging
from statistics import mean, median
from typing import Any

import sglang.bench_one_batch as bench


def main() -> None:
    version = importlib.metadata.version("sglang")
    if version != "0.5.9":
        raise RuntimeError(f"Recheck the benchmark adapter for SGLang {version}")
    parser = argparse.ArgumentParser(description=__doc__)
    bench.ServerArgs.add_cli_args(parser)
    bench.BenchArgs.add_cli_args(parser)
    parser.add_argument("--comparison-warmup", type=int, default=2)
    parser.add_argument("--comparison-iterations", type=int, default=3)
    args = parser.parse_args()
    if args.comparison_warmup < 1 or args.comparison_iterations < 1:
        parser.error("warmup and iterations must be positive")
    prepare = bench.prepare_synthetic_inputs_for_latency_test
    measure = bench.latency_test_run_once

    # Dynamic values are confined to the installed upstream benchmark boundary.
    def fixed_inputs(batch_size: int, input_len: int, custom_inputs: Any = None) -> Any:
        if custom_inputs:
            raise ValueError("Use deterministic token inputs, not --prompt-filename")
        return prepare(batch_size, input_len, [[9707] * input_len for _ in range(batch_size)])

    def repeated_measure(*call_args: Any, **call_kwargs: Any) -> dict[str, Any] | None:
        output_len = call_args[6]
        samples: list[dict[str, Any]] = []
        for iteration in range(args.comparison_warmup + args.comparison_iterations):
            positional = list(call_args)
            # Req objects are mutated by extend/decode; regenerate them each time.
            positional[3] = fixed_inputs(positional[4], positional[5])
            result = measure(*positional, **call_kwargs)
            if result is None:
                return None
            if iteration >= args.comparison_warmup:
                samples.append(result)
        return {
            "sglang_version": version,
            "dtype": args.dtype,
            "cuda_graph": not args.disable_cuda_graph,
            "batch_size": call_args[4],
            "input_len": call_args[5],
            "output_len": output_len,
            "warmup": args.comparison_warmup,
            "iterations": args.comparison_iterations,
            "prefill_mean_seconds": mean(s["prefill_latency"] for s in samples),
            "decode_median_seconds": (
                median(s["median_decode_latency"] for s in samples)
                if output_len > 1
                else None
            ),
            "samples": samples,
        }

    bench.prepare_synthetic_inputs_for_latency_test = fixed_inputs
    bench.latency_test_run_once = repeated_measure
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bench.main(bench.ServerArgs.from_cli_args(args), bench.BenchArgs.from_cli_args(args))


if __name__ == "__main__":
    main()
