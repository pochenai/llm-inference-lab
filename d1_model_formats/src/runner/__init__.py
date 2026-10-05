"""Select a backend without importing unrelated execution libraries."""

from types import TracebackType
from typing import Protocol, Self

from numpy.typing import ArrayLike

from ..common import ARTIFACTS, Backend, FloatArray, PathLike, StorageMode


class Runner(Protocol):
    """The shared interface; backends need not inherit from a common base class."""

    def predict(self, x: ArrayLike) -> FloatArray: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


def load_runner(
    backend: Backend, directory: PathLike = ARTIFACTS, mode: StorageMode = "mmap",
) -> Runner:
    # ONNX and ggml inference do not import PyTorch or its model definition.
    if backend == "torch":
        from .torch import TorchRunner
        return TorchRunner(directory)
    if backend == "onnx":
        from .onnx import OnnxRunner
        return OnnxRunner(directory)
    if backend == "ggml":
        from .ggml import GGMLRunner
        return GGMLRunner(directory, mode)
    raise ValueError(f"Unknown backend: {backend}")
