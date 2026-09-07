"""Load GGUF in Python, construct the FFN graph, and execute real ggml CPU ops."""

from pathlib import Path
from types import TracebackType
from typing import Self

from numpy.typing import ArrayLike

from ..common import Config, FloatArray, PathLike, StorageMode
from ..model.gguf_ggml import run_gguf_ggml
from ..storage import WeightFile
from ._ggml import CData, GGML, Pointer

# ggml context capacity stores tensor metadata and optionally tensor data. This
# fixed reservation contains only no-alloc weight tensor descriptors, not weights.
WEIGHT_CONTEXT_BYTES: int = 64 * 1024
# The CPU backend in the pinned ggml revision requires an aligned host pointer.
CPU_BACKING_ALIGNMENT_BYTES: int = 64
class GGMLRunner:
    def __init__(
        self, directory: PathLike, mode: StorageMode = "mmap", *,
        library_dir: PathLike | None = None,
    ) -> None:
        self.lib = GGML(library_dir)
        self.storage: WeightFile | None = None
        self.config: Config | None = None
        self.weight_context: CData | None = None
        self.buffer: CData | None = None
        self.tensors: dict[str, CData] = {}
        self.closed = False
        try:
            self.storage = WeightFile(Path(directory) / "model.gguf", mode)
            self.config = self.storage.config
            if self.config is None:
                raise ValueError("GGUF is missing TinyFFN metadata")
            self.weight_context = self.lib.context(WEIGHT_CONTEXT_BYTES, no_alloc=True)
            raw = self.storage.raw
            assert raw is not None
            if raw.ctypes.data % CPU_BACKING_ALIGNMENT_BYTES:
                raise ValueError(f"ggml CPU backing storage must be {CPU_BACKING_ALIGNMENT_BYTES}-byte aligned")
            # This non-owning buffer wraps mmap pages or bytes filled by readinto().
            self.buffer = self.lib.base.ggml_backend_cpu_buffer_from_ptr(self.lib.ffi.from_buffer(raw), raw.nbytes)
            if self.buffer == self.lib.ffi.NULL:
                raise MemoryError("ggml CPU buffer creation failed")
            for name, array in self.storage.weights.items():
                if array.ndim == 1:
                    tensor = self.lib.base.ggml_new_tensor_1d(
                        self.weight_context, self.lib.base.GGML_TYPE_F32, array.shape[0],
                    )
                else:
                    # NumPy [out, in] and ggml ne=[in, out] share identical bytes.
                    tensor = self.lib.base.ggml_new_tensor_2d(
                        self.weight_context, self.lib.base.GGML_TYPE_F32, array.shape[1], array.shape[0],
                    )
                if tensor == self.lib.ffi.NULL:
                    raise MemoryError(f"ggml tensor allocation failed: {name}")
                status = self.lib.base.ggml_backend_tensor_alloc(
                    self.buffer, tensor, self.lib.ffi.from_buffer(array),
                )
                if status != 0:
                    raise RuntimeError(f"ggml weight attachment failed: {name} (status {status})")
                self.tensors[name] = tensor
        except Exception:
            self.close()
            raise

    def predict(self, x: ArrayLike) -> FloatArray:
        if self.closed:
            raise RuntimeError("Runner is closed")
        assert self.config is not None
        return run_gguf_ggml(self.lib, self.config, self.tensors, x)

    def weight_addresses(self) -> dict[str, Pointer]:
        if self.closed:
            raise RuntimeError("Runner is closed")
        return {name: self.lib.address(self.lib.base.ggml_get_data(tensor)) for name, tensor in self.tensors.items()}

    def close(self) -> None:
        if self.closed:
            return
        self.tensors.clear()
        if self.weight_context is not None:
            self.lib.base.ggml_free(self.weight_context)
            self.weight_context = None
        if self.buffer is not None:
            self.lib.base.ggml_backend_buffer_free(self.buffer)
            self.buffer = None
        if self.storage:
            self.storage.close()
            self.storage = None
        self.config = None
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
