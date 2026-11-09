# Rejected B1-only vector shared expert

Operator tiny/real/masked-tail fixtures match the unfused reference bitwise;
15 BF16 regressions pass. However, real 35B chunking differs by 0.31143212 in
logits, above the existing 0.02 threshold. Full prefill still matches the prior
path (reference max 0.6661885, RMS 0.12629366). No native performance samples
were run and no production commit was made. Do not widen the threshold.

The one-token full-K reduction and multi-token MMA reduction can round
 differently before BF16 conversion; the model amplifies those differences.
Try consistent arithmetic across token counts, then validate and measure.
