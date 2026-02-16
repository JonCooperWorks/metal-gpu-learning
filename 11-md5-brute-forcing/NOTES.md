# Lesson 11 – MD5 brute-forcing notes

## MD5 auxiliary functions (F, G, H, I)

- Names come from **RFC 1321** (MD5 spec). Keeping `F`, `G`, `H`, `I` is the standard convention.
- **F** `(x & y) | (~x & z)` — “if x then y else z” (selection).
- **G** `(x & z) | (y & ~z)` — “if z then x else y”.
- **H** `x ^ y ^ z` — parity (XOR of the three).
- **I** `y ^ (x | ~z)` — mixing for round 4.
- Used one per round: round 1 → F, round 2 → G, round 3 → H, round 4 → I.

## GPU branching

- The match check (`a == target_a && ...`) and `params.mode` branch in the kernel are **fine**:
  - In brute-force, almost no threads find a match → almost everyone takes the same path (no divergence on the hot path).
  - `params.mode` is uniform (same for all threads) → no divergence.
  - Divergence only when a thread actually finds a match, which is rare.
- **When to worry:** branching is a problem when a **large fraction** of threads (e.g. ~50%) take different sides of a condition → warps serialize both paths.
- **Mantra:** Coherent threads = happy GPU.

## When ~50/50: different kernels / “hashmap vibes”

- If the split is roughly half-and-half and the branch is expensive, consider **separate kernels** instead of one branchy kernel:
  1. **Classify** — One pass: each thread writes which path it takes (e.g. 0 or 1) and maybe its index.
  2. **Stream compact / partition** — Build two index buffers: “threads that take path A” and “threads that take path B”.
  3. **Launch separate kernels** — Kernel A runs only on path-A indices, kernel B only on path-B indices. Each kernel then has coherent threads (no divergence).
- Same idea as “group by outcome, then process each group” — no actual hash map needed, just buffers and two kernels.

## Summary

- **Rare branch (e.g. MD5 match):** Keep the `if`; divergence cost is negligible.
- **~50/50 and expensive branch:** Classify → partition → one kernel per path.
