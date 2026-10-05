"""Mapped views and eagerly read storage for the FFN fixture files."""

from dataclasses import asdict
import json
import mmap
from pathlib import Path
import struct
from types import TracebackType
from typing import Protocol, Self, TypedDict, cast

import gguf
import numpy as np

from .common import ByteArray, Config, PathLike, StorageMode, Weights


class MappedBytes(Protocol):
    """The mapping owner exposed by NumPy memmap, absent from NumPy's stubs."""

    _mmap: mmap.mmap


class TensorHeader(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


def aligned_read(path: Path) -> ByteArray:
    """Read a file directly into independently owned, 64-byte-aligned bytes."""
    size = path.stat().st_size
    owner = np.empty(size + 63, dtype=np.uint8)
    offset = (-owner.ctypes.data) % 64
    result = owner[offset:offset + size]
    with path.open("rb") as file:
        bytes_read = file.readinto(memoryview(result))
    if bytes_read != size:
        raise OSError(f"Could not read the complete file: {path}")
    return result


class WeightFile:
    """Keep views and their backing storage alive together; views expire on close."""

    def __init__(self, path: PathLike, mode: StorageMode = "mmap") -> None:
        if mode not in ("mmap", "read"):
            raise ValueError("Storage mode must be mmap or read")
        self.path, self.mode = Path(path), mode
        self.weights: Weights = {}
        self.offsets: dict[str, int] = {}
        self.disk_shapes: dict[str, tuple[int, ...]] = {}
        self.reader: gguf.GGUFReader | None = None
        self.mapping: mmap.mmap | None = None
        self.raw: ByteArray | None = None
        self.config: Config | None = None
        self.closed = False
        try:
            if self.path.suffix == ".gguf":
                self._load_gguf()
            elif self.path.suffix == ".safetensors":
                self._load_safetensors()
            else:
                raise ValueError("Expected a .gguf or .safetensors file")
        except Exception:
            self.close()
            raise

    def _load_gguf(self) -> None:
        reader = gguf.GGUFReader(self.path, mode="r")
        if reader.byte_order != "I":
            raise ValueError("This experiment requires native-endian GGUF tensors")
        config: dict[str, object] = {}
        for key in asdict(Config()):
            field_name = "general.architecture" if key == "architecture" else f"tiny_ffn.{key}"
            field = reader.get_field(field_name)
            if field is None:
                raise ValueError(f"Missing GGUF metadata: {field_name}")
            config[key] = field.contents()
        self.config = Config.from_dict(config)
        if self.mode == "mmap":
            self.mapping = cast(MappedBytes, reader.data)._mmap
            self.raw = reader.data
            # Keep the reader alive because its memmap owns the mapped bytes.
            self.reader = reader
        else:
            self.raw = aligned_read(self.path)
        for tensor in reader.tensors:
            if tensor.name in self.weights:
                raise ValueError(f"Duplicate tensor: {tensor.name}")
            if tensor.tensor_type != gguf.GGMLQuantizationType.F32:
                raise ValueError(f"{tensor.name}: only F32 GGUF tensors are supported")
            # GGUF ne[] lists the fastest-varying dimension first.
            shape = tuple(int(n) for n in reversed(tensor.shape))
            self.offsets[tensor.name] = tensor.data_offset
            self.disk_shapes[tensor.name] = tuple(int(n) for n in tensor.shape)
            self.weights[tensor.name] = np.ndarray(shape, np.float32, buffer=self.raw, offset=tensor.data_offset)

    def _load_safetensors(self) -> None:
        # ACCESS_COPY keeps the source unchanged and allows safe torch.from_numpy views.
        # Clean pages remain file-backed; actual writes would trigger copy-on-write.
        if self.mode == "mmap":
            with self.path.open("rb") as file:
                self.mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
            self.raw = np.ndarray((len(self.mapping),), np.uint8, buffer=self.mapping)
        else:
            self.raw = aligned_read(self.path)
        assert self.raw is not None
        if self.raw.nbytes < 8:
            raise ValueError("Truncated Safetensors header")
        header_size = struct.unpack_from("<Q", self.raw)[0]
        data_start = 8 + header_size
        if header_size > 100_000_000 or data_start > self.raw.nbytes:
            raise ValueError("Invalid Safetensors header size")
        header: dict[str, object] = json.loads(bytes(self.raw[8:data_start]))
        ranges: list[tuple[int, int]] = []
        for name, raw_entry in header.items():
            if name == "__metadata__":
                continue
            # JSON is untyped; the dtype, dimensions, and offsets are checked below.
            entry = cast(TensorHeader, raw_entry)
            shape = tuple(entry["shape"])
            begin, end = entry["data_offsets"]
            if entry["dtype"] != "F32" or any(type(n) is not int or n < 0 for n in shape):
                raise ValueError(f"{name}: expected a valid F32 tensor")
            size = int(np.prod(shape)) * 4
            if begin < 0 or end < begin or end - begin != size or data_start + end > self.raw.nbytes:
                raise ValueError(f"{name}: invalid data offsets")
            offset = data_start + begin
            self.offsets[name], self.disk_shapes[name] = offset, shape
            self.weights[name] = np.ndarray(shape, "<f4", buffer=self.raw, offset=offset)
            ranges.append((begin, end))
        position = 0
        for begin, end in sorted(ranges):
            if begin != position:
                raise ValueError("Safetensors data ranges must not overlap or contain holes")
            position = end
        if data_start + position != self.raw.nbytes:
            raise ValueError("Unindexed Safetensors payload bytes")

    def shares_mapping(self, name: str) -> bool:
        if self.closed:
            raise RuntimeError("Weight file is closed")
        if self.mapping is None:
            return False
        mapped = np.ndarray((len(self.mapping),), np.uint8, buffer=self.mapping)
        return bool(np.shares_memory(self.weights[name], mapped))

    def close(self) -> None:
        if not self.closed:
            self.weights.clear()
            self.raw = self.reader = None
            if self.mapping is not None:
                self.mapping.close()
                self.mapping = None
            self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
