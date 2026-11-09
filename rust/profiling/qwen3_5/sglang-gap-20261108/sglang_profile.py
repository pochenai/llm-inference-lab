"""Profile the unchanged SGLang benchmark after warmup; hooks are eager-only."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any

import sglang.bench_one_batch as bench
import torch

OUT = Path(os.environ["QWEN35_PROFILE_DIR"])
EAGER = "--disable-cuda-graph" in sys.argv
active = False
sample = 0
completed = False
record_ids = False
warm_ids: list[list[int]] = []
measure = bench.latency_test_run_once


def profiled_measure(*args: Any, **kwargs: Any) -> Any:
    global active, sample, completed, record_ids
    # Upstream's initial warmup uses eight positional arguments and keyword options.
    assert len(args) in (8, 15), "SGLang benchmark signature changed"
    is_sweep = len(args) == 15
    capture = is_sweep and sample == 2
    record_ids = is_sweep and sample == 1
    if record_ids:
        warm_ids.clear()
    if capture:
        assert args[4:7] == (1, 32, 8)
        torch.cuda.synchronize()
        torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
        active = True
    try:
        return measure(*args, **kwargs)
    finally:
        if capture:
            torch.cuda.synchronize()
            active = False
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
            completed = True
        if record_ids:
            (OUT / ("sglang-eager-ids.json" if EAGER else "sglang-graph-ids.json")).write_text(json.dumps(warm_ids) + "\n")
        record_ids = False
        if is_sweep:
            sample += 1


def phase_wrapper(function: Any, label: str) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if active:
            torch.cuda.nvtx.range_push(label)
        try:
            result = function(*args, **kwargs)
            if record_ids:
                # Warmup only: do not add token readbacks to the profiled graph path.
                warm_ids.append(result[0].tolist())
            return result
        finally:
            if active:
                torch.cuda.nvtx.range_pop()
    return wrapped


def hooks(module: Any, label: str) -> None:
    def push(_module: Any, _inputs: Any) -> None:
        torch.cuda.nvtx.range_push(label)
    def pop(_module: Any, _inputs: Any, _output: Any) -> None:
        torch.cuda.nvtx.range_pop()
    module.register_forward_pre_hook(push)
    module.register_forward_hook(pop, always_call=True)


load = bench.load_model


def load_and_describe(*args: Any, **kwargs: Any) -> Any:
    loaded = load(*args, **kwargs)
    modules: list[dict[str, Any]] = []
    for name, module in loaded[0].model.named_modules():
        record: dict[str, Any] = {"name": name, "class": type(module).__name__}
        for attribute in ("weight", "w13_weight", "w2_weight"):
            weight = getattr(module, attribute, None)
            if isinstance(weight, torch.Tensor):
                record[attribute] = {"shape": list(weight.shape), "dtype": str(weight.dtype)}
        quant = getattr(module, "quant_method", None)
        if quant is not None:
            record["quant_method"] = type(quant).__name__
        modules.append(record)
        if EAGER and ("Linear" in type(module).__name__ or "MoE" in type(module).__name__ or name.rsplit(".", 1)[-1] in {"linear_attn", "self_attn"}):
            hooks(module, "sglang.module." + name)
    (OUT / ("sglang-eager-modules.json" if EAGER else "sglang-graph-modules.json")).write_text(json.dumps(modules, indent=2) + "\n")
    return loaded


bench.load_model = load_and_describe
bench.latency_test_run_once = profiled_measure
bench.extend = phase_wrapper(bench.extend, "sglang.prefill")
bench.decode = phase_wrapper(bench.decode, "sglang.decode")
runpy.run_path("rust/engine/benches/sglang_qwen2.py", run_name="__main__")
assert completed and sample == 3, (completed, sample)
