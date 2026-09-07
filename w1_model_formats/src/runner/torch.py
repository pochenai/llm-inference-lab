"""Load config and Safetensors weights into the separately defined model."""

from pathlib import Path
from types import TracebackType
from typing import Self

import torch
from numpy.typing import ArrayLike
from safetensors.torch import load_file

from ..common import Config, FloatArray, PathLike, as_input
from ..model.torch import MODEL_REGISTRY, TinyFFN


class TorchRunner:
    def __init__(self, directory: PathLike) -> None:
        directory = Path(directory)
        config = Config.load(directory / "config.json")
        weights = load_file(str(directory / "model.safetensors"), device="cpu")
        model = MODEL_REGISTRY[config.architecture](config)
        # Default assign=False copies values into existing Parameter storage.
        model.load_state_dict(weights, strict=True)
        self.model: TinyFFN | None = model.eval()

    def predict(self, x: ArrayLike) -> FloatArray:
        if self.model is None:
            raise RuntimeError("Runner is closed")
        with torch.inference_mode():
            return self.model(torch.from_numpy(as_input(x))).numpy()

    def close(self) -> None:
        self.model = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
