# Repository conventions

## Organize experiments by roadmap week

- This repository contains multiple weeks of the inference learning roadmap.
  W1 is one experiment, not the entire project.
- Keep all week-specific source code, tests, notes, documentation, fixtures,
  generated artifacts, and native build dependencies inside that week's directory.
- W1 lives in `w1_model_formats/`, with Python code in `src/`, tests in `tests/`,
  usage instructions in `README.md`, outputs in `artifacts/`, and ggml in `.deps/`.
- Use the same `w<number>_<topic>/` organization for later weeks. Do not put new
  week-specific code or tests at the repository root.
- Keep shared tooling at the root: `pyproject.toml`, `uv.lock`, `.python-version`,
  `.gitignore`, this file, and a short README linking to the weekly experiments.
- Separate source responsibilities: `src/model/torch.py` defines the PyTorch
  FFN, `src/model/gguf_ggml.py` builds the ggml computation graph recovered from
  GGUF weights, `src/runner/` owns format loading and resource lifetimes, and
  `src/cmd/` contains CLI entry points. `src/common.py` holds shared contracts;
  `src/storage.py` owns mapped weight storage; keep low-level ggml bindings in
  `runner/_ggml.py`.
- Keep experiment READMEs self-contained: explain the model, code structure,
  and runnable commands without assuming knowledge of the weekly roadmap.
- Use fully qualified imports such as `w1_model_formats.src.common`. Run commands
  from the repository root, for example `python -m w1_model_formats.src.cmd.predict`.
  Avoid a global `src` import that could conflict with another week's package.
- When moving files, update imports, subprocess module names, pytest discovery,
  resource paths, documentation links, and CLI examples together. Keep generated
  outputs and third-party source/build directories ignored by Git.

## Language and environment

- Write new code, comments, docstrings, CLI arguments, CLI messages, and project
  documentation in English. Preserve existing user-authored notes unless asked
  to edit or translate them.
- Annotate Python function parameters and return values, including tests. Use
  explicit types for public interfaces, mutable containers, and optional resource
  state; let obvious local values be inferred. Use NumPy dtype-aware array aliases
  and a shared runner protocol. Keep dynamic typing confined to validated external
  boundaries such as JSON and CFFI rather than spreading `Any` through callers.
- Run `env -u LD_LIBRARY_PATH uv run --offline pyright` after type-related changes.
- Use uv and the shared root environment. The project requires Python `>=3.12`;
  `.python-version` selects the existing `3.12.3` interpreter. Keep automatic
  Python downloads disabled and reuse existing dependency caches.
- W1 uses CPU-only PyTorch and ggml. Do not add CUDA packages for this experiment.
- On this host, an inherited `LD_LIBRARY_PATH` may point to another Python
  environment's PyTorch libraries. Use `env -u LD_LIBRARY_PATH uv run ...` when
  needed, rather than modifying the user's shell configuration.

## Validation

- Run the affected week's tests after code or layout changes. For W1:
  `env -u LD_LIBRARY_PATH uv run --offline pytest -q w1_model_formats/tests`.
- Keep the root pytest configuration updated when adding another week's tests.
- Preserve unrelated user changes, and do not claim verification without running it.

### GPU stream ordering tests

- Prefer a `Gate::arm(&stream)` test helper over sleeps or elapsed-time assertions
  when testing submission ordering, in-flight resource ownership, or independent
  completion. It enqueues a host callback that waits on a channel: earlier stream
  work can finish, while later work stays blocked until the gate is released.
- Reference implementation: `../cuda-rust/cutile-rs/cutile/tests/gpu/submission_lifetimes.rs`.
  `Gate` is a local test helper, not a public cuda-async API. Reuse an existing
  helper or adapt the minimal pattern; callbacks must not call CUDA APIs.
- Warm up kernel compilation/module loading and required allocations before
  arming a gate: context-wide synchronization during initialization can otherwise
  wait on the test's own blocked stream. Engine GPU tests use `GPU_TEST_LOCK`.
- Release the gate before awaiting or dropping a future that drains the blocked
  stream on the same thread; use an independent releaser when testing that drain.
  Give the callback a bounded wait and a Drop-based release path so failures do
  not hang indefinitely. A Tokio timeout cannot interrupt a blocking poll/drop.
- Assert ordering and ownership directly: a gated future remains Pending; releasing
  the first gate lets its future complete while a later gate still holds its
  submission/resources. Detect gate expiration separately and fail diagnostically;
  an expired gate must not make an ordering test pass.

## Secret handling

- Never read, print, search, summarize, quote, diff, copy, or otherwise access real
  secret environment files, including `.env`, `*.env`, `.env.local`,
  `.env.development`, `.env.production`, `.env.test`, `.env.staging`, or their
  `.local` variants. Exclude these files from searches and directory-wide reads.
- `.env.example` is allowed as a non-secret example template.
- If a task requires values from a real environment file, ask the user to provide
  the needed value manually or move it into a safe non-secret fixture first.
