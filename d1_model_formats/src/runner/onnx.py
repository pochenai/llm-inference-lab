"""Execute the graph stored in ONNX without importing the PyTorch model class."""

from pathlib import Path
from types import TracebackType
from typing import Self, cast

from numpy.typing import ArrayLike
import onnxruntime as ort

from ..common import FloatArray, PathLike, as_input


class OnnxRunner:
    def __init__(self, directory: PathLike) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session: ort.InferenceSession | None = ort.InferenceSession(
            str(Path(directory) / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def predict(self, x: ArrayLike) -> FloatArray:
        if self.session is None:
            raise RuntimeError("Runner is closed")
        # ONNX serializes f itself, so this runner has no Python model module to import.
        # This exported graph has one FP32 tensor output, not a sequence or map.
        return cast(FloatArray, self.session.run(["logits"], {"x": as_input(x)})[0])

    def close(self) -> None:
        self.session = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
