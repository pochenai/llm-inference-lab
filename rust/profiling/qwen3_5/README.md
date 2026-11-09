# Qwen3.5 BF16 benchmark evidence

Start with the [machine handoff and iteration log](../../engine/task.md).
Work is paused because the source machine has no GPU. Production code is
committed through `827f7f9`; expert tile16 was rejected and reverted. MoE branch
parallelism has only a passing CUDA dependency prototype, not inference support.

Latest native B1/32-input/8-output Rust graph snapshot: prefill32.499ms,
decode7.705ms/token, E2E86.433ms. Earlier SGLang graph:69.120/7.049/119.244ms.
Decode parity is outstanding; final same-period cross-engine confirmation is
pending. Per-iteration tables are paired comparisons, not cross-round subtraction.

## Evidence map

| Directory | Content |
| --- | --- |
| `cutile-0.5.0-20261109/` | Migration baseline and validation |
| `cuda-graph-20261108/` | Original graph comparison and SGLang configuration |
| `sglang-gap-20261108/` | Paired profiling and MoE overlap ablation |
| `replay-storage-20261108/` | Scratch storage deduplication |
| `expert-mma-20261108/` | BF16 expert MMA, raw GPU code/SASS evidence |
| `shared-gate-up-20261109/` | Shared fusion; failed B1-only variant retained |
| `router-vectors-20261109/` | Accepted scalar/router vector kernels |
| `skinny-projections-20261109/` | Rejected all-vector a/b and precision screen |
| `packed-ab-20261109/` | Accepted precision-preserving a/b packing |
| `frozen-weights-20261109/` | Capture-owned immutable weights and lifetime checks |
| `post-packed-profile-20261109/`, `post-frozen-profile-20261109/` | CPU attribution before/after frozen weights |
| `blocked-gqa-20261109/` | Blocked attention and old/new/fenced memcheck logs |
| `finite-scan-20261109/` | Latest accepted complete CPU finite scan |
| `expert-tile16-20261109/` | Rejected tile16 performance screen |
| `moe-parallel-20261109/probe/` | Passing child-node/dependency insertion prototype |
| `validation-references/` | Small reference logits and SHA256 manifest |
| `reference-notes/` | Copied Qwen2 Tensor Core precedent used during this work |

Reports, raw JSON/JSONL/logs, commands, source snapshots/diffs, plots, SASS,
small extracted GPU-code images and Nsight reports are versioned.
SQLite exports and sidecars are ignored; derive them without a GPU when needed:

```bash
nsys export --type=sqlite --output=/path/to/report.sqlite /path/to/report.nsys-rep
```

Host executables named baseline/candidate/etc and temporary jit-store caches
remain ignored. Binary hashes identify the original runs; rebuild the appropriate
commit or recorded candidate patch on the replacement machine. Historical scripts
contain old absolute paths and should write new directories when rerun.

Full model weights, generated tiny weights, Python virtual environments and
build outputs are not included. Transfer/regenerate these separately. The CUDA
and GPU architecture, CPU NUMA affinity, SGLang/dependency versions and performance
tiles must be checked afresh; old numbers/CPU24-31 are not new-machine guarantees.

`parallel-design.md` is an unimplemented proposal. `frozen-weights-plan.md` records
historical reasoning for the implemented private weight type. Preserve rejected
experiments as evidence, not as alternate production defaults.
