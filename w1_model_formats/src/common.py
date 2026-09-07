"""Shared data contract; importing this module does not import PyTorch."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

type PathLike = str | Path
type FloatArray = NDArray[np.float32]
type ByteArray = NDArray[np.uint8]
type Weights = dict[str, FloatArray]
type Backend = Literal["torch", "ggml", "onnx"]
type StorageMode = Literal["mmap", "read"]

ARTIFACTS: Path = Path(__file__).resolve().parents[1] / "artifacts"
SHAPES: dict[str, tuple[int, ...]] = {
    "fc1.weight": (8, 4), "fc1.bias": (8,), "fc2.weight": (3, 8), "fc2.bias": (3,),
}


@dataclass(frozen=True)
class Config:
    architecture: str = "tiny_ffn"
    input_dim: int = 4
    hidden_dim: int = 8
    output_dim: int = 3
    activation: str = "relu"
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: object) -> "Config":
        expected = asdict(cls())
        if not isinstance(data, dict) or data != expected or any(
            type(data[k]) is not type(v) for k, v in expected.items()
        ):
            raise ValueError(f"Unsupported model config: expected {expected}, got {data}")
        # Validation accepts exactly this experiment's config, including value types.
        return cls()

    @classmethod
    def load(cls, path: PathLike) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def make_weights() -> Weights:
    rng = np.random.default_rng(42)
    return {name: (rng.standard_normal(shape) * 0.25).astype(np.float32) for name, shape in SHAPES.items()}


def as_input(x: ArrayLike) -> FloatArray:
    value = np.asarray(x, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 4 or value.shape[0] < 1:
        raise ValueError("Input must have shape [batch, 4] with a nonempty batch")
    if not np.isfinite(value).all():
        raise ValueError("Input must contain finite values")
    return np.ascontiguousarray(value)


def reference(x: ArrayLike, weights: Weights) -> FloatArray:
    """A NumPy reference used only for verification, never as a ggml fallback."""
    h = np.maximum(as_input(x) @ weights["fc1.weight"].T + weights["fc1.bias"], 0)
    return h @ weights["fc2.weight"].T + weights["fc2.bias"]


def test_inputs() -> dict[str, FloatArray]:
    return {
        "single": np.array([[1, 2, 3, 4]], dtype=np.float32),
        "zero": np.zeros((1, 4), dtype=np.float32),
        "mixed": np.array([[-2, 1, -3, 0.5], [4, -1, 0, -2]], dtype=np.float32),
        "batch": np.random.default_rng(123).standard_normal((7, 4)).astype(np.float32),
    }
