# patricia_tree 0.10.2 dependency review

Reviewed on 2026-11-01 before adding the dependency. No malicious indicators were
found in the checks below. This is a scoped supply-chain/source review, not proof
of absence of malware or a formal audit of every unsafe memory operation.

| Check | Result |
| --- | --- |
| Distribution | crates.io `patricia_tree` **0.10.2**, pinned exactly in Cargo.toml and Cargo.lock |
| Archive SHA-256 | `7fd106b8571932f2b8d961a296390c56f05d454ce95e2405f1d8703ca0119e8e`; matches crates.io version metadata and Cargo.lock |
| Upstream source | All six packaged `src/*.rs` files match commit `f959e447722792af0bd891a16376e1cd591f4bb9` in `sile/patricia_tree` |
| Features/dependencies | `default-features = false`; serde disabled; `cargo tree -p patricia_tree --edges normal` lists no runtime dependencies |
| Build-time execution | `build = false`, no build script, proc-macro target, native linking or assembly directives found |
| Side-effect scan | No network, process spawning, environment-variable or filesystem access found in runtime source |
| Unsafe code | Located in node layout/allocation, pointer access, value movement, Send/Sync and Drop; reviewed relevant allocation/access/release paths |
| Known advisories | OSV query for crates.io `patricia_tree@0.10.2` returned no vulnerabilities on the review date |
| Tool limitation | cargo-audit was not installed; it was not run. No Miri or formal memory-safety verification was performed. |
| License/MSRV | MIT; Rust 1.85 minimum, below this workspace's 1.89 requirement |

The package denies unsafe code at crate scope but explicitly permits it inside
node.rs. It must not be described as an entirely safe-Rust implementation.
Queries/inserts use capped byte slices; no unbounded full prompt is passed to the
library. The existing prefix oracle, UTF-8 byte-cap and removal tests exercise
our adapter. Re-review source and advisories before changing the pinned version.

Evidence: the archive checksum, upstream comparisons and OSV response are saved
in the ignored local `rust/profiling/routing/patricia-20261101/security-scan.json`.
The release API probe was tested separately before integration.

Sources: [published version metadata](https://crates.io/api/v1/crates/patricia_tree/0.10.2),
[pinned upstream source](https://github.com/sile/patricia_tree/tree/f959e447722792af0bd891a16376e1cd591f4bb9/src),
[OSV query API](https://google.github.io/osv.dev/api/#tag/vulnerability/operation/queryVulnerability).

## Recursive-depth boundary and adapter guard

An isolated debug build of the upstream map overflowed a 2 MiB thread stack with
4,096 nested prefixes (the release probe passed). This is an availability issue,
not evidence of malicious behavior. A prefix-length cap alone is insufficient:
same-length keys with deeply branching paths can also create deep recursion.

The integration therefore keeps at most 128 keys in each PatriciaMap, splitting
at 129 keys using standard BTreeMap range fences. Lookup checks at most three
neighbor buckets, preserving exact longest-common-prefix semantics even after
removal. Configuration rejects prefix caps above 64 KiB; the default remains
4 KiB. This also bounds the library's 255-byte label-chunk recursion. Fence keys
and bucket metadata are included in the conservative index charge.

The debug adapter test passes on a 2 MiB stack with 4,096 nested prefixes,
4,096 equal-length branching keys, removals, and 300 maximum-size keys. A separate
independent prefix-scan oracle checks bucket splits and removals. These are bounded
regression tests, not a formal guarantee for arbitrary upstream changes.
