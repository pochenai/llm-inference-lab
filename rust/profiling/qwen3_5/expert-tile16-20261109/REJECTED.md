# Rejected expert output tile16 screen

BF16 MMA-only output tile32->16, M16/K64 unchanged; SIMT/f32 remains32.
CPU/SIMT oracle and real reference pass (max0.47784/RMS0.08379/ID3636,
all four chunks max0). One-process screen,2 warmups+3 samples:
prefill69.405ms, decode8.331ms/token, e2e127.722ms. This is materially slower
than the preceding7.705ms/decode,86.433ms/e2e baseline. No confirmation or
production commit. Source/binary/hash/commands and raw screen retained.
Qwen2 N16 rejection also cautioned against assuming more CTAs always help;
no hardware-counter cause is claimed. Production code restored to tile32.
