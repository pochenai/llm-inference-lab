"""Load one saved model and predict from four input numbers."""

import argparse
import json
from pathlib import Path

from ..common import ARTIFACTS
from ..runner import load_runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "ggml", "onnx"], required=True)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    parser.add_argument("--storage", choices=["mmap", "read"], help="GGUF backing storage (default: mmap)")
    parser.add_argument("--x", type=float, nargs=4, required=True, metavar=("X1", "X2", "X3", "X4"))
    args = parser.parse_args()
    if args.storage is not None and args.backend != "ggml":
        parser.error("--storage applies only to the ggml backend; see mmap_demo for Safetensors")
    try:
        with load_runner(args.backend, args.directory, args.storage or "mmap") as runner:
            logits = runner.predict([args.x])
        print(json.dumps({"backend": args.backend, "logits": logits.tolist(), "prediction": logits.argmax(axis=-1).tolist()}, indent=2))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
