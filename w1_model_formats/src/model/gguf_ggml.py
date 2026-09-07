"""Build and execute TinyFFN's function from GGUF weights using ggml CPU ops."""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike

from ..common import Config, FloatArray, as_input
from ..runner._ggml import CData, GGML

# Graph capacity is a maximum node count. This FFN creates five operator nodes;
# 64 leaves room for graph bookkeeping and is not a model dimension.
GRAPH_NODE_CAPACITY: int = 64
# A fixed thread count keeps this tiny educational run deterministic and avoids
# selecting a host-dependent default. It is a runtime policy, not model metadata.
CPU_THREAD_COUNT: int = 1
# ggml's context owns activation tensors and compute scratch. The base is
# conservative fixed headroom; per-row bytes below come from this exact FFN graph.
ACTIVATION_CONTEXT_BASE_BYTES: int = 1024 * 1024
F32_BYTES: int = np.dtype(np.float32).itemsize


def activation_context_bytes(config: Config, batch_size: int) -> int:
    """Return capacity for x, affine intermediates, ReLU, and logits."""
    elements_per_row = config.input_dim + 3 * config.hidden_dim + 2 * config.output_dim
    return ACTIVATION_CONTEXT_BASE_BYTES + batch_size * elements_per_row * F32_BYTES


def run_gguf_ggml(
    lib: GGML, config: Config, weights: Mapping[str, CData], x: ArrayLike,
) -> FloatArray:
    """Recover f in Python: Gemm, add, ReLU, Gemm, add, then execute it with ggml."""
    values = as_input(x)
    context = lib.context(activation_context_bytes(config, values.shape[0]), no_alloc=False)
    try:
        inputs = lib.base.ggml_new_tensor_2d(
            context, lib.base.GGML_TYPE_F32, config.input_dim, values.shape[0],
        )
        lib.ffi.memmove(lib.base.ggml_get_data(inputs), lib.ffi.from_buffer(values), values.nbytes)

        hidden = lib.base.ggml_mul_mat(context, weights["fc1.weight"], inputs)
        hidden = lib.base.ggml_add(context, hidden, weights["fc1.bias"])
        hidden = lib.base.ggml_relu(context, hidden)
        logits = lib.base.ggml_mul_mat(context, weights["fc2.weight"], hidden)
        logits = lib.base.ggml_add(context, logits, weights["fc2.bias"])

        graph = lib.base.ggml_new_graph_custom(context, GRAPH_NODE_CAPACITY, False)
        lib.base.ggml_build_forward_expand(graph, logits)
        status = lib.cpu.ggml_graph_compute_with_ctx(context, graph, CPU_THREAD_COUNT)
        if status != 0:
            raise RuntimeError(f"ggml graph execution failed (status {status})")
        pointer = lib.base.ggml_get_data(logits)
        output = lib.ffi.buffer(pointer, values.shape[0] * config.output_dim * F32_BYTES)
        return np.frombuffer(output, dtype=np.float32).reshape(-1, config.output_dim).copy()
    finally:
        # Output above is copied before this call-local activation context is freed.
        lib.base.ggml_free(context)
