"""Generate CFFI declarations from the pinned ggml public headers.

This is a small adaptation of ggml's examples/python/regenerate.py. It omits
optional Python stubs and accepts duplicate typedefs present in this pinned
header set. No C function signatures are maintained in Python.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import cffi


def compiler_flags(include_dir: Path, fake_headers: Path) -> list[str]:
    return [
        "-I", str(include_dir),
        "-I", str(fake_headers),
        "-D__GNUC_VA_LIST",
        "-D__gnuc_va_list=void*",
        "-D__fp16=uint16_t",
        "-D__attribute__(x)=",
        "-D_Static_assert(x, m)=",
    ]


def preprocess(compiler: str, flags: list[str], api_header: Path) -> str:
    result = subprocess.run(
        [compiler, "-E", *flags, str(api_header)], check=True, capture_output=True, text=True,
    )
    return "\n".join(line for line in result.stdout.splitlines() if "__darwin_va_list" not in line)


def evaluate_size_expressions(compiler: str, flags: list[str], api_header: Path, header: str) -> str:
    expressions = set(re.findall(r"(?<=\[)[^\]]+(?=\])|sizeof\s*\([^()]+\)", header))
    with tempfile.TemporaryDirectory(prefix="ggml-cffi-") as directory:
        executable = Path(directory) / "eval_size_expr"
        for expression in expressions:
            if re.fullmatch(r"\d+|\s*", expression):
                continue
            program = (
                "#include <stdio.h>\n"
                f"#include \"{api_header}\"\n"
                f"int main() {{ printf(\"%lu\", (size_t)({expression})); }}\n"
            )
            subprocess.run(
                [compiler, "-o", str(executable), *flags, "-x", "c", "-"],
                check=True, input=program, text=True,
            )
            size = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout
            header = header.replace(expression, size)
    return header


def generate(api_header: Path, include_dir: Path, fake_headers: Path, output_dir: Path) -> Path:
    compiler = os.environ.get("CC", "gcc")
    flags = compiler_flags(include_dir, fake_headers)
    header = preprocess(compiler, flags, api_header)
    header = evaluate_size_expressions(compiler, flags, api_header, header)
    builder: Any = cffi.FFI()
    builder.cdef(header, override=True)
    builder.set_source("_w1_ggml_cffi", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder.compile(tmpdir=str(output_dir), verbose=True)
    return output_dir / "_w1_ggml_cffi.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", type=Path, required=True)
    parser.add_argument("--include", type=Path, required=True)
    parser.add_argument("--fake-headers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = generate(args.api, args.include, args.fake_headers, args.output)
    print(f"Generated CFFI binding: {binding}")


if __name__ == "__main__":
    main()
