"""Check serialization, real execution, storage ownership, and error contracts."""

import ast
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
import pytest
from safetensors.numpy import load_file, save_file
import torch

from w1_model_formats.src.common import (
    Backend,
    Config,
    SHAPES,
    StorageMode,
    as_input,
    make_weights,
    reference,
    test_inputs as sample_inputs,
)
from w1_model_formats.src.cmd.export import export_models
from w1_model_formats.src.model.gguf_ggml import (
    ACTIVATION_CONTEXT_BASE_BYTES,
    F32_BYTES,
    activation_context_bytes,
)
from w1_model_formats.src.runner.ggml import GGMLRunner
from w1_model_formats.src.cmd.mmap_demo import demonstrate, digest
from w1_model_formats.src.runner._ggml import GGML
from w1_model_formats.src.runner import load_runner
from w1_model_formats.src.storage import WeightFile
from w1_model_formats.src.model.torch import TinyFFN
from w1_model_formats.src.runner.torch import TorchRunner


@pytest.fixture(scope="session")
def models(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    export_models(path)
    return path


def test_all_formats_preserve_weights(models: Path) -> None:
    expected = make_weights()
    assert sum(value.size for value in expected.values()) == 67
    assert sum(value.nbytes for value in expected.values()) == 268
    onnx_model = onnx.load(models / "model.onnx")
    onnx.checker.check_model(onnx_model)
    assert [node.op_type for node in onnx_model.graph.node] == ["Gemm", "Relu", "Gemm"]
    assert [
        node.attribute[0].i for node in onnx_model.graph.node if node.op_type == "Gemm"
    ] == [1, 1]
    initializers = {
        value.name: onnx.numpy_helper.to_array(value)
        for value in onnx_model.graph.initializer
    }
    safe = load_file(str(models / "model.safetensors"))
    with WeightFile(models / "model.gguf") as gguf:
        assert gguf.disk_shapes["fc1.weight"] == (4, 8)
        assert gguf.weights["fc1.weight"].shape == (8, 4)
        for name in SHAPES:
            for actual in (safe[name], initializers[name], gguf.weights[name]):
                np.testing.assert_array_equal(actual, expected[name])


@pytest.mark.parametrize(
    "backend,mode",
    [("torch", "mmap"), ("onnx", "mmap"), ("ggml", "mmap"), ("ggml", "read")],
)
@pytest.mark.parametrize("case", list(sample_inputs()))
def test_predictions_and_lifetime(
    models: Path, backend: Backend, mode: StorageMode, case: str,
) -> None:
    x = sample_inputs()[case]
    expected = reference(x, make_weights())
    runner = load_runner(backend, models, mode)
    with runner:
        first = runner.predict(x)
        np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(first.argmax(axis=-1), expected.argmax(axis=-1))
        for _ in range(3):
            np.testing.assert_array_equal(runner.predict(x), first)
    runner.close()
    np.testing.assert_allclose(first, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(RuntimeError, match="closed"):
        runner.predict(x)


@pytest.mark.parametrize("mode", ["mmap", "read"])
@pytest.mark.parametrize("filename", ["model.safetensors", "model.gguf"])
def test_file_storage_sharing(models: Path, filename: str, mode: StorageMode) -> None:
    path = models / filename
    before = digest(path)
    with WeightFile(path, mode) as store:
        assert store.raw is not None
        for name in SHAPES:
            assert store.shares_mapping(name) is (mode == "mmap")
            assert (
                store.weights[name].ctypes.data
                == store.raw.ctypes.data + store.offsets[name]
            )
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.shares_mapping("fc1.weight")
    assert digest(path) == before


@pytest.mark.parametrize("mode", ["mmap", "read"])
def test_ggml_uses_the_storage_address(models: Path, mode: StorageMode) -> None:
    path = models / "model.gguf"
    before = digest(path)
    with GGMLRunner(models, mode) as runner:
        assert runner.storage is not None
        addresses = runner.weight_addresses()
        for name in SHAPES:
            assert addresses[name] == runner.storage.weights[name].ctypes.data
        runner.predict([[1, 2, 3, 4]])
    assert digest(path) == before


def test_load_state_dict_copies_parameters(models: Path) -> None:
    with WeightFile(models / "model.safetensors") as store:
        model = TinyFFN(Config())
        state = {name: torch.from_numpy(value) for name, value in store.weights.items()}
        model.load_state_dict(state)
        for name, param in model.named_parameters():
            assert state[name].data_ptr() == store.weights[name].ctypes.data
            assert param.data_ptr() != state[name].data_ptr()
            np.testing.assert_array_equal(param.detach().numpy(), store.weights[name])
        state.clear()


@pytest.mark.parametrize(
    "change",
    [
        {"architecture": "unknown"},
        {"activation": "gelu"},
        {"hidden_dim": 9},
        {"input_dim": 4.0},
    ],
)
def test_config_rejects_incompatible_models(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Unsupported model config"):
        Config.from_dict(asdict(Config()) | change)


@pytest.mark.parametrize("fault", ["missing", "shape"])
def test_invalid_weights_are_reported(tmp_path: Path, fault: str) -> None:
    weights = make_weights()
    if fault == "missing":
        del weights["fc1.bias"]
    else:
        weights["fc1.weight"] = np.zeros((4, 8), dtype=np.float32)
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(asdict(Config())))
    with pytest.raises(RuntimeError, match="Missing key|size mismatch"):
        TorchRunner(tmp_path)


@pytest.mark.parametrize(
    "x", [[], [1, 2, 3, 4], [[1, 2, 3]], [[1, 2, 3, float("nan")]]]
)
def test_invalid_input(x: list[float] | list[list[float]]) -> None:
    with pytest.raises(ValueError, match="Input must"):
        as_input(x)


def test_missing_native_library_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="build_ggml"):
        GGML(library_dir=tmp_path)


@pytest.mark.parametrize("backend", ["torch", "ggml", "onnx"])
def test_standalone_prediction(models: Path, backend: Backend) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "w1_model_formats.src.cmd.predict",
            "--backend",
            backend,
            "--directory",
            str(models),
            "--x",
            "1",
            "2",
            "3",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    np.testing.assert_allclose(
        data["logits"], reference([[1, 2, 3, 4]], make_weights()), rtol=1e-5, atol=1e-6
    )
    assert data["prediction"] == [0]


def test_onnx_loads_without_python_architecture_or_config(models: Path, tmp_path: Path) -> None:
    # Isolate the ONNX file: the config and Safetensors file are absent.
    import shutil

    shutil.copyfile(models / "model.onnx", tmp_path / "model.onnx")
    code = """
import sys
from w1_model_formats.src.runner import load_runner
with load_runner('onnx', sys.argv[1]) as runner:
    assert runner.predict([[1, 2, 3, 4]]).shape == (1, 3)
assert 'torch' not in sys.modules
assert 'w1_model_formats.src.runner.torch' not in sys.modules
assert 'w1_model_formats.src.model' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)


def test_mmap_demo(models: Path, capsys: pytest.CaptureFixture[str]) -> None:
    demonstrate(models)
    assert "source file hashes are unchanged" in capsys.readouterr().out


def test_runtime_is_cpu_only() -> None:
    assert torch.version.cuda is None
    assert sys.version_info >= (3, 12)


def test_activation_context_size_is_dimension_derived() -> None:
    config = Config()
    expected_elements = config.input_dim + 3 * config.hidden_dim + 2 * config.output_dim
    assert activation_context_bytes(config, 2) == (
        ACTIVATION_CONTEXT_BASE_BYTES + 2 * expected_elements * F32_BYTES
    )


def test_function_signatures_are_annotated() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").rglob("*.py"))
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            location = f"{path.relative_to(root)}:{node.lineno} {node.name}"
            assert node.returns is not None, f"Missing return type: {location}"
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg is not None:
                arguments.append(node.args.vararg)
            if node.args.kwarg is not None:
                arguments.append(node.args.kwarg)
            for argument in arguments:
                if argument.arg not in {"self", "cls"}:
                    assert argument.annotation is not None, (
                        f"Missing parameter type: {location}: {argument.arg}"
                    )
