"""Verify all backends against the same reference, including repeat inference."""

import argparse
from pathlib import Path

import numpy as np

from ..common import ARTIFACTS, Backend, PathLike, StorageMode, make_weights, reference, test_inputs
from ..runner import load_runner


def compare(directory: PathLike = ARTIFACTS) -> None:
    weights = make_weights()
    print("backend       case       max_abs_error   prediction")
    variants: tuple[tuple[Backend, StorageMode], ...] = (
        ("torch", "mmap"), ("onnx", "mmap"), ("ggml", "mmap"), ("ggml", "read"),
    )
    for backend, mode in variants:
        with load_runner(backend, directory, mode) as runner:
            for name, x in test_inputs().items():
                expected = reference(x, weights)
                actual = runner.predict(x)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                np.testing.assert_array_equal(actual.argmax(axis=-1), expected.argmax(axis=-1))
                np.testing.assert_array_equal(runner.predict(x), actual)
                label = f"ggml/{mode}" if backend == "ggml" else backend
                print(f"{label:13} {name:10} {np.max(np.abs(actual - expected)):13.6g}   {actual.argmax(axis=-1).tolist()}")
    print("PASS: every backend meets rtol=1e-5 / atol=1e-6 and agrees on predictions.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    compare(parser.parse_args().directory)


if __name__ == "__main__":
    main()
