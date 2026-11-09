# Archived reference logits

Files contain 248,320 little-endian float32 logits (993,280 bytes each). They are
small numerical outputs, not model weights. SHA256/size/original paths are in
manifest.json. Default reference tests use prompt IDs [1,7,9,3,12].

Use `real-reference.f32` with the original Qwen3.5-35B-A3B checkpoint and
`tiny-reference.f32` with the matching generated tiny fixture. The other real
files preserve earlier diagnostic/reference variants; they are not replacement
acceptance defaults. New hardware/library versions may require independently
regenerating and examining reference differences; do not silently replace or
widen old acceptance gates.

Set QWEN35_TEST_MODEL and QWEN35_TEST_REFERENCE to absolute paths. See
rust/engine/task.md and rust/engine/examples/qwen3_5.md for test/fixture commands.
