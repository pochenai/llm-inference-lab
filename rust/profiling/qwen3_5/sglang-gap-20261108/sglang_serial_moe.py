"""Diagnostic control: serialize only shared/routed MoE branches before capture."""
import runpy
from typing import Any
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

original_init = Qwen2MoeSparseMoeBlock.__init__


def serial_init(self: Any, *args: Any, **kwargs: Any) -> None:
    original_init(self, *args, **kwargs)
    self.alt_stream = None


Qwen2MoeSparseMoeBlock.__init__ = serial_init
runpy.run_path('rust/engine/benches/sglang_qwen2.py', run_name='__main__')
