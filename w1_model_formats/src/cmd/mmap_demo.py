"""Compare mapped weights, eager file reads, and runtime parameter storage."""

import argparse
import hashlib
import mmap
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
import torch

from ..common import ARTIFACTS, Config, PathLike, StorageMode, reference, test_inputs
from ..runner.ggml import GGMLRunner
from ..storage import WeightFile
from ..model.torch import MODEL_REGISTRY


def digest(path: PathLike) -> str:
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def demonstrate(directory: PathLike = ARTIFACTS) -> None:
    directory = Path(directory)
    paths = [directory / name for name in ("model.safetensors", "model.gguf")]
    before = {path: digest(path) for path in paths}
    official = load_file(str(paths[0]))
    x = test_inputs()["mixed"]
    expected = reference(x, official)
    print(f"OS page size: {mmap.PAGESIZE} bytes")
    print("These tiny files may each fit in one page. This is a storage experiment, not a speed benchmark.")
    print("mmap keeps file-backed views; read uses ordinary file.readinto() into owned memory.")
    print("Safetensors uses private copy-on-write mapping; GGUF uses a read-only mapping.")

    modes: tuple[StorageMode, ...] = ("mmap", "read")
    for mode in modes:
        print(f"\nSafetensors, storage={mode}")
        with WeightFile(paths[0], mode) as store:
            config = Config.load(directory / "config.json")
            model = MODEL_REGISTRY[config.architecture](config).eval()
            state = {name: torch.from_numpy(value) for name, value in store.weights.items()}
            model.load_state_dict(state)
            params = dict(model.named_parameters())
            for name, value in store.weights.items():
                # This validates every loaded FP32 value, not merely its header shape.
                np.testing.assert_array_equal(value, official[name])
                shared = store.shares_mapping(name)
                copied = state[name].data_ptr() != params[name].data_ptr()
                assert shared == (mode == "mmap")
                assert copied
                # The print below is observational: shape/offset/address alone do
                # not prove correct values or correct inference. The assertions
                # above and below provide those checks.
                print(f"{name}: offset={store.offsets[name]}, shape={value.shape}, dtype={value.dtype}, "
                      f"view=0x{value.ctypes.data:x}, parameter=0x{params[name].data_ptr():x}, "
                      f"shares_mapping={shared}, load_state_dict_copied={copied}")
            with torch.inference_mode():
                actual = model(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            state.clear()  # Release torch views before the mmap owner closes.
        print(f"prediction={actual.argmax(axis=-1).tolist()}")

        print(f"\nGGUF + ggml, storage={mode}")
        with GGMLRunner(directory, mode) as runner:
            assert runner.storage is not None
            addresses = runner.weight_addresses()
            for name, value in runner.storage.weights.items():
                # This validates every GGUF tensor value against the official loader.
                np.testing.assert_array_equal(value, official[name])
                shared = runner.storage.shares_mapping(name)
                assert shared == (mode == "mmap")
                assert addresses[name] == value.ctypes.data
                # This print observes storage ownership; it is not the value check.
                print(f"{name}: offset={runner.storage.offsets[name]}, shape={value.shape}, dtype={value.dtype}, "
                      f"view=0x{value.ctypes.data:x}, ggml=0x{addresses[name]:x}, shares_mapping={shared}")
            actual = runner.predict(x)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        print(f"prediction={actual.argmax(axis=-1).tolist()}")

    for path in paths:
        assert digest(path) == before[path], f"Source file changed: {path}"
    print("\nPASS: outputs agree, storage sharing is verified, and source file hashes are unchanged.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ARTIFACTS)
    demonstrate(parser.parse_args().directory)


if __name__ == "__main__":
    main()
