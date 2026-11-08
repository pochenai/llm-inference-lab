# BF16 text-engine baseline validation — 2026-11-08

Validated on one NVIDIA RTX 6000D (85,651 MiB), CUDA driver 595.91.07,
CUDA 13.2, cuTile Rust 0.4.0 and an existing PyTorch 2.8.0+cu128 environment.
This is the standalone engine baseline, not completed W10 acceptance.

## Checkpoint and memory

The config, index, tokenizer and all fourteen shard download metadata files
identify Hugging Face revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
This revision identifies the original validation run; configuration and weights
are read directly from the supplied model directory.

The loader measured **69,321,574,912 resident weight bytes** (64.56 GiB),
excluding vision and MTP. A 4096-token context needs **148,275,200 bytes**
for attention K/V, convolution history and FP32 recurrent state, excluding
scratch and allocator/JIT overhead.

## Numerical checks

The separate eager PyTorch reference loads checkpoint weights directly and
uses no Rust operators. Its BF16 projections keep partial reductions in FP32;
router ties choose the lowest expert index. Probe IDs: `[1,7,9,3,12]`.

| Model | Maximum absolute logit error | RMS logit error | Greedy ID (Rust/reference) |
| --- | ---: | ---: | --- |
| Reduced four-layer fixture | 2.3841858e-7 | 2.409685e-8 | 157831 / 157831 |
| Real 40-layer checkpoint | 0.6103344 | 0.121414535 | 3636 / 3636 |

For **both models**, chunk sizes 1/2/3/4 produced **zero maximum logit
difference** against full prefill and identical next-token IDs. The one-token
case exercises recurrent and K/V continuation.

The reduced fixture requires maximum error <0.005 and RMS <0.0005. The real
BF16 baseline requires maximum error <1.0, relative RMS <5% and matching probe
argmax; measured relative RMS is about 4.52%. This broader budget is provisional:
BF16 rounding and discrete expert selection amplify differences across layers.
It is not bitwise equivalence or a held-out quality evaluation. Default PyTorch
reduced-precision BF16 accumulation selected a different greedy ID on this
ambiguous probe; backend precision modes must be stated in comparisons.

A dedicated convolution test checks all three history planes directly for
chunk sizes 1/2/5, including `[0,0,1]` after the first token and `[3,4,5]` after
five tokens. It guards a cuTile loop-carry save issue that collapsed all three
history planes; saving the raw projection suffix independently fixes it.
A separate GPU test checks deterministic tied expert indices and normalized
scores, including large negative router logits. Config and input checks cover
unsupported FP8/features, context capacity, invalid IDs and changed histories.

## Executed checks

- Qwen3.5 release suite with the real checkpoint: **5 passed**.
- Reduced-fixture reference/chunking release test: **1 passed**.
- Existing Qwen2 packed/ragged/history/decode regression: **1 passed**.
- `cargo check --manifest-path rust/Cargo.toml --offline -p engine --all-targets`: passed.
- `cargo fmt --manifest-path rust/Cargo.toml --all --check`: passed.
- `python -m py_compile rust/engine/examples/qwen3_5_reference.py`: passed.
- An assert-based fixture-path check confirmed generation rejects directories
  outside the ignored `rust/engine/artifacts/qwen3_5/` and preserves an existing sentinel file.
- Required `env -u LD_LIBRARY_PATH uv run --offline pyright`: blocked because
  the shared environment's locked `torch==2.14.0+cpu` wheel is absent from its cache.

Reproduction commands are in [example guide](qwen3_5.md).

## Text smoke test and remaining limits

The release example loaded the real checkpoint, formatted a user turn with
thinking disabled, and answered `What is 2 + 2? Answer with one digit.` with
**`4`**, then stopped at EOS before its four-token budget. The prompt had 25
tokens and the context allocation budget was 64,983,040 bytes.
Cold measurements were 48.18 s loading, 13.568 s prefill and 2.545 s decode;
these include initialization/JIT and are **not warmed latency/throughput results**.

Runner/CausalModel integration, packed batching, prefix snapshots/reuse, GPU
sampling, Graphs, workspace reuse, long-context/held-out quality and comparison
against a full Transformers/SGLang pipeline remain pending. A Gate-based
cancellation probe expired during submission initialization and was not accepted
as a passing ownership/ordering test. The current path can block on initialization
and allocations; no submission-first overlap or warmed gated-cancellation claim
is made. Context poisoning and retained GPU allocation owners are implemented,
but that cancellation stress coverage still needs workspace/input reuse.
