"""Fetch pinned ggml, build CPU libraries, and generate CFFI bindings from headers."""

import argparse
import subprocess
from pathlib import Path

from ..runner._ggml import BUILD, GGML_COMMIT, SOURCE
from ..runner.generate_cffi import generate

API_HEADER: Path = Path(__file__).resolve().parents[1] / "runner" / "ggml_api.h"
FAKE_HEADERS: Path = API_HEADER.parent / "cffi_fake_headers"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def generate_bindings() -> None:
    """Use ggml's generator; the local header selects the public APIs we need."""
    generate(API_HEADER, SOURCE / "include", FAKE_HEADERS, BUILD / "python")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use existing source only")
    parser.add_argument("--jobs", type=int, default=2, help="Number of build workers")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not (SOURCE / ".git").exists():
        if args.offline:
            parser.error("Pinned ggml source is missing; run once without --offline")
        SOURCE.mkdir(parents=True, exist_ok=True)
        run("git", "init", str(SOURCE))
        run("git", "-C", str(SOURCE), "remote", "add", "origin", "https://github.com/ggml-org/ggml.git")
    current = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    if current.stdout.strip() != GGML_COMMIT:
        if args.offline:
            parser.error("Existing ggml checkout does not match the pinned revision")
        run("git", "-C", str(SOURCE), "fetch", "--depth=1", "origin", GGML_COMMIT)
        run("git", "-C", str(SOURCE), "checkout", "--detach", GGML_COMMIT)
    run(
        "cmake", "-S", str(SOURCE), "-B", str(BUILD),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF",
        "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF",
        "-DGGML_BLAS=OFF", "-DGGML_OPENMP=OFF", "-DGGML_NATIVE=OFF",
    )
    run("cmake", "--build", str(BUILD), "--parallel", str(args.jobs))
    generate_bindings()
    print(f"ggml revision: {GGML_COMMIT}\nLibraries: {BUILD / 'src'}\nBindings: generated from {API_HEADER}")


if __name__ == "__main__":
    main()
