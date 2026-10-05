"""Save the same deterministic FFN parameters in three real model formats."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import gguf
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file

from ..common import ARTIFACTS, Config, PathLike, Weights, make_weights


def export_models(directory: PathLike = ARTIFACTS) -> Weights:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = make_weights()
    config: dict[str, str | int] = asdict(Config())
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(weights, str(directory / "model.safetensors"))

    writer = gguf.GGUFWriter(str(directory / "model.gguf"), "tiny_ffn")
    for key, value in config.items():
        if key == "architecture":
            continue  # GGUFWriter already writes general.architecture.
        if isinstance(value, int):
            writer.add_uint32(f"tiny_ffn.{key}", value)
        else:
            writer.add_string(f"tiny_ffn.{key}", value)
    for name, tensor in weights.items():
        # The writer reverses dimension metadata; the row-major bytes are unchanged.
        writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    nodes = [
        helper.make_node("Gemm", ["x", "fc1.weight", "fc1.bias"], ["preactivation"], transB=1),
        helper.make_node("Relu", ["preactivation"], ["h"]),
        helper.make_node("Gemm", ["h", "fc2.weight", "fc2.bias"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "TinyFFN",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 3])],
        initializer=[numpy_helper.from_array(value, name) for name, value in weights.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    weights = export_models(args.directory)
    print(f"Saved {sum(w.size for w in weights.values())} FP32 parameters (268 payload bytes)")
    for name in ("config.json", "model.safetensors", "model.gguf", "model.onnx"):
        path = args.directory / name
        print(f"{path.name:20} {path.stat().st_size:6} bytes  {path}")


if __name__ == "__main__":
    main()
