"""Load CFFI bindings generated from the pinned ggml public headers."""

import ctypes
import importlib.util
from pathlib import Path
import sys
from typing import Any, cast

from ..common import PathLike

type CData = Any
type CFFI = Any
type Pointer = int

GGML_COMMIT: str = "e91ded11bdcd78c42f9c8d3978ff6686eb4c1226"
SOURCE: Path = Path(__file__).resolve().parents[2] / ".deps" / "ggml"
BUILD: Path = SOURCE / "build"
BINDING_FILE: Path = BUILD / "python" / "_w1_ggml_cffi.py"


class GGML:
    """Two library handles backed by generated, revision-matched CFFI declarations."""

    def __init__(self, library_dir: PathLike | None = None) -> None:
        directory = Path(library_dir) if library_dir is not None else BUILD / "src"
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        self.ffi = self._load_ffi()
        try:
            # ggml-cpu resolves symbols from ggml-base. ctypes supplies the
            # RTLD_GLOBAL loading mode that CFFI's dlopen wrapper does not expose.
            self._base_dependency = ctypes.CDLL(
                str(directory / f"libggml-base{suffix}"), mode=ctypes.RTLD_GLOBAL,
            )
            self._cpu_dependency = ctypes.CDLL(str(directory / f"libggml-cpu{suffix}"))
            self.base = self.ffi.dlopen(str(directory / f"libggml-base{suffix}"))
            self.cpu = self.ffi.dlopen(str(directory / f"libggml-cpu{suffix}"))
        except OSError as error:
            raise RuntimeError(
                "Could not load ggml. Run: uv run python -m w1_model_formats.src.cmd.build_ggml\n"
                f"Library directory: {directory}\n{error}"
            ) from error
        commit = self.ffi.string(self.base.ggml_commit()).decode()
        revision = commit.removesuffix("-dirty")
        if len(revision) < 7 or not GGML_COMMIT.startswith(revision):
            raise RuntimeError(f"ggml revision mismatch: expected {GGML_COMMIT}, got {commit}")

    @staticmethod
    def _load_ffi() -> CFFI:
        if not BINDING_FILE.exists():
            raise RuntimeError(
                "Could not find generated ggml bindings. "
                "Run: uv run python -m w1_model_formats.src.cmd.build_ggml"
            )
        specification = importlib.util.spec_from_file_location("_w1_ggml_cffi", BINDING_FILE)
        if specification is None or specification.loader is None:
            raise RuntimeError(f"Could not load generated ggml bindings: {BINDING_FILE}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        return cast(CFFI, module.ffi)

    def context(self, size: int, *, no_alloc: bool) -> CData:
        params = self.ffi.new(
            "struct ggml_init_params *",
            {"mem_size": size, "mem_buffer": self.ffi.NULL, "no_alloc": no_alloc},
        )
        context = self.base.ggml_init(params[0])
        if context == self.ffi.NULL:
            raise MemoryError("ggml context allocation failed")
        return context

    def address(self, pointer: CData) -> Pointer:
        return int(self.ffi.cast("uintptr_t", pointer))
