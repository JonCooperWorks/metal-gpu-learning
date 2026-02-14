# GPU Programming with Metal in Rust

A hands-on tutorial for learning Apple's Metal GPU compute API through Rust.

Each lesson is a self-contained Cargo crate in its own subfolder, progressively
introducing Metal concepts from device discovery to parallel compute kernels.

## Prerequisites

- macOS (Metal is Apple-only)
- Rust toolchain (`rustup` / `cargo`)
- Xcode Command Line Tools (`xcode-select --install`)

## Lessons

| Lesson | Folder | What you'll learn |
|--------|--------|-------------------|
| 1 | `01-hello-gpu/` | Device discovery -- connect to the GPU, query its properties |
| 2 | `02-double-values/` | Your first compute kernel -- the full Metal pipeline end-to-end |
| 3 | `03-vector-add/` | Multiple input buffers -- element-wise vector addition |
| 4 | `04-parallel-sum/` | Atomic operations -- parallel reduction to sum an array |
| 5 | `05-double-buffering/` | Overlap CPU fill and GPU compute with two alternating buffer slots |
| 6 | `06-market-tick-aggregation/` | Real analytics pipeline -- transform market ticks and reduce to VWAP/volatility/min/max |
| 7 | `07-credit-card-fraud-detector/` | Fraud rules engine -- score transactions and reduce TP/FP/TN/FN + precision/recall/FPR metrics |
| 8 | `08-basic-llm/` | Basic LLM inference -- single-head attention + tiny autoregressive generation loop |

## How to Run

From the workspace root:

```bash
# Run a specific lesson
cargo run -p hello-gpu

# Or cd into the folder
cd 01-hello-gpu
cargo run

# Build everything at once
cargo build --workspace
```

## How to Read the Code

Every `src/main.rs` is written as a tutorial. Read top-to-bottom -- the comments
explain every concept as it's introduced. The Metal Shading Language (MSL) source
code is embedded as Rust string constants so you can see the GPU and CPU code
side by side.
