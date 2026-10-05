"""Show where tensor bytes and model computation are stored."""

import argparse
import json
from pathlib import Path

import onnx

from ..common import ARTIFACTS
from ..storage import WeightFile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    print("Safetensors config (selects Python architecture code):")
    print(json.dumps(json.loads((args.directory / "config.json").read_text()), indent=2))
    for name in ("model.safetensors", "model.gguf"):
        with WeightFile(args.directory / name) as store:
            print(f"\n{name}: {store.path.stat().st_size} bytes; no serialized operator graph")
            if name.endswith(".gguf"):
                print(f"GGUF config: {store.config}")
            print("tensor          dtype    numpy_shape  disk_shape   offset  bytes")
            for key, array in store.weights.items():
                print(f"{key:15} {str(array.dtype):8} {str(array.shape):12} "
                      f"{str(store.disk_shapes[key]):12} {store.offsets[key]:6} {array.nbytes:6}")
    model = onnx.load(args.directory / "model.onnx")
    print(f"\nmodel.onnx: IR {model.ir_version}, opsets {[(x.domain, x.version) for x in model.opset_import]}")
    for tensor in model.graph.initializer:
        print(f"initializer {tensor.name}: shape={list(tensor.dims)}, payload={len(tensor.raw_data)} bytes")
    for node in model.graph.node:
        attributes = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        print(f"{node.op_type}({', '.join(node.input)}) -> {', '.join(node.output)} {attributes}")
    print("ONNX initializers are Protobuf fields; they do not use the GGUF/Safetensors offset table.")


if __name__ == "__main__":
    main()
