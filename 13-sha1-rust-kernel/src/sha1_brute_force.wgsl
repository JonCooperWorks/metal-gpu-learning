// WGSL port of lesson 12's Metal kernel: sha1_brute_force.metal
// The algorithm and host/kernel contract are intentionally aligned with the MSL version.

// Host-written dispatch parameters.
// Equivalent to MSL struct KernelParams.
struct KernelParams {
    // Candidate length for this dispatch pass.
    len: u32,
    // Radix for diagnostics (26, 36, 95). Mapping uses alphabet_id branches.
    radix: u32,
    // Total candidates for current length, i.e. radix^len.
    search_space: u64,
    // Per-thread work chunk in the inner loop.
    candidates_per_thread: u32,
    // Match mode: 0 = first match, 1 = all matches.
    mode: u32,
    // Capacity guard for all-matches mode.
    max_matches: u32,
    // Alphabet selector: 0=lower, 1=lowernum, 2=printable.
    alphabet_id: u32,
    // Target SHA1 digest words (big-endian word interpretation).
    target_a: u32,
    target_b: u32,
    target_c: u32,
    target_d: u32,
    target_e: u32,
    // Padding to keep host/kernel layout stable.
    _pad0: u32,
}

// Shared "found" flag for first-match mode.
// Equivalent to MSL buffer(1) atomic_uint found_flag.
struct FoundFlagBuffer {
    value: atomic<u32>,
}

// Winning index for first-match mode.
// Equivalent to MSL buffer(2) ulong found_index.
struct FoundIndexBuffer {
    value: u64,
}

// Atomic count of discovered matches in all-matches mode.
// Equivalent to MSL buffer(3) atomic_uint match_count.
struct MatchCountBuffer {
    value: atomic<u32>,
}

// Output slots storing candidate indices that matched target hash.
// Equivalent to MSL buffer(4) ulong* match_indices.
struct MatchIndicesBuffer {
    values: array<u64>,
}

// SHA1 digest words returned by sha1_one_block.
// Equivalent to MSL's a,b,c,d,e output references.
struct Digest5 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    e: u32,
}

// Result of custom 64-bit / 32-bit division routine.
// Used so mapping can process u64 index values with explicit q/rem control.
struct DivMod64By32 {
    q_hi: u32,
    q_lo: u32,
    rem: u32,
}

// WGSL binding model for buffers.
// MSL counterpart uses [[buffer(n)]].
@group(0) @binding(0) var<storage, read> params: KernelParams;
@group(0) @binding(1) var<storage, read_write> found_flag: FoundFlagBuffer;
@group(0) @binding(2) var<storage, read_write> found_index: FoundIndexBuffer;
@group(0) @binding(3) var<storage, read_write> match_count: MatchCountBuffer;
@group(0) @binding(4) var<storage, read_write> match_indices: MatchIndicesBuffer;

// Workgroup size override populated from host pipeline constants.
// Equivalent operational role to threads-per-threadgroup in MSL dispatch.
override WORKGROUP_SIZE: u32 = 256u;

// Rotate-left helper used by SHA1 round and schedule logic.
// Equivalent to MSL inline rotl().
fn rotl(x: u32, s: u32) -> u32 {
    return (x << s) | (x >> (32u - s));
}

// Read 4 bytes from message block (byte array in u32 lanes) as big-endian u32.
// Equivalent to MSL load_be_u32().
fn load_be_u32(block: ptr<function, array<u32, 64>>, off: u32) -> u32 {
    return ((*block)[off] << 24u)
        | ((*block)[off + 1u] << 16u)
        | ((*block)[off + 2u] << 8u)
        | (*block)[off + 3u];
}

// SHA1 round function selector by t range.
// Equivalent to MSL sha1_f().
fn sha1_f(t: u32, b: u32, c: u32, d: u32) -> u32 {
    // Rounds 0..19: choose function.
    if (t < 20u) {
        return (b & c) | ((~b) & d);
    }
    // Rounds 20..39: parity.
    if (t < 40u) {
        return b ^ c ^ d;
    }
    // Rounds 40..59: majority.
    if (t < 60u) {
        return (b & c) | (b & d) | (c & d);
    }
    // Rounds 60..79: parity.
    return b ^ c ^ d;
}

// SHA1 additive constants by round region.
// Equivalent to MSL sha1_k().
fn sha1_k(t: u32) -> u32 {
    if (t < 20u) {
        return 0x5A827999u;
    }
    if (t < 40u) {
        return 0x6ED9EBA1u;
    }
    if (t < 60u) {
        return 0x8F1BBCDCu;
    }
    return 0xCA62C1D6u;
}

// Divide 64-bit unsigned integer (n_hi:n_lo) by 32-bit d.
// Returns quotient split in q_hi:q_lo and remainder rem.
// MSL can do ulong div/mod directly in map_* helpers; this helper keeps WGSL path explicit.
fn divmod_u64_by_u32(n_hi: u32, n_lo: u32, d: u32) -> DivMod64By32 {
    // Remainder carried across 16-bit limbs.
    var rem: u32 = 0u;

    // Split numerator into 4x16-bit chunks from high to low.
    let a3 = n_hi >> 16u;
    let a2 = n_hi & 0xffffu;
    let a1 = n_lo >> 16u;
    let a0 = n_lo & 0xffffu;

    // Long-division step for highest limb.
    let t3 = (rem << 16u) | a3;
    let q3 = t3 / d;
    rem = t3 - q3 * d;

    // Next limb.
    let t2 = (rem << 16u) | a2;
    let q2 = t2 / d;
    rem = t2 - q2 * d;

    // Next limb.
    let t1 = (rem << 16u) | a1;
    let q1 = t1 / d;
    rem = t1 - q1 * d;

    // Lowest limb.
    let t0 = (rem << 16u) | a0;
    let q0 = t0 / d;
    rem = t0 - q0 * d;

    // Reassemble 64-bit quotient and return remainder.
    var out: DivMod64By32;
    out.q_hi = (q3 << 16u) | q2;
    out.q_lo = (q1 << 16u) | q0;
    out.rem = rem;
    return out;
}

// Map candidate index -> lowercase ASCII candidate bytes.
// Equivalent to MSL map_lower().
fn map_lower(idx: u64, len: u32, out: ptr<function, array<u32, 55>>) {
    // Mutable quotient state (64-bit split into hi/lo words).
    var q_hi = u32(idx >> 32u);
    var q_lo = u32(idx & u64(0xffffffffu));

    // Emit len digits in base-26 (little-endian digit order by position i).
    var i: u32 = 0u;
    loop {
        if (i >= len) {
            break;
        }
        // Extract next radix digit using custom u64 div/mod.
        let dm = divmod_u64_by_u32(q_hi, q_lo, 26u);
        // 'a' + digit.
        (*out)[i] = 97u + dm.rem;
        // Continue with quotient for next character position.
        q_hi = dm.q_hi;
        q_lo = dm.q_lo;
        i = i + 1u;
    }
}

// Map candidate index -> lowercase + digits ASCII candidate bytes.
// Equivalent to MSL map_lowernum().
fn map_lowernum(idx: u64, len: u32, out: ptr<function, array<u32, 55>>) {
    var q_hi = u32(idx >> 32u);
    var q_lo = u32(idx & u64(0xffffffffu));

    var i: u32 = 0u;
    loop {
        if (i >= len) {
            break;
        }
        // Base-36 digit extraction.
        let dm = divmod_u64_by_u32(q_hi, q_lo, 36u);
        // 0..25 -> 'a'..'z'.
        if (dm.rem < 26u) {
            (*out)[i] = 97u + dm.rem;
        } else {
            // 26..35 -> '0'..'9'.
            (*out)[i] = 48u + (dm.rem - 26u);
        }
        q_hi = dm.q_hi;
        q_lo = dm.q_lo;
        i = i + 1u;
    }
}

// Map candidate index -> printable ASCII [0x20..0x7e] candidate bytes.
// Equivalent to MSL map_printable().
fn map_printable(idx: u64, len: u32, out: ptr<function, array<u32, 55>>) {
    var q_hi = u32(idx >> 32u);
    var q_lo = u32(idx & u64(0xffffffffu));

    var i: u32 = 0u;
    loop {
        if (i >= len) {
            break;
        }
        // Base-95 digit extraction.
        let dm = divmod_u64_by_u32(q_hi, q_lo, 95u);
        // Offset into printable ASCII region.
        (*out)[i] = 32u + dm.rem;
        q_hi = dm.q_hi;
        q_lo = dm.q_lo;
        i = i + 1u;
    }
}

// Single-block SHA1 implementation for len <= 55 bytes.
// Equivalent to MSL sha1_one_block().
fn sha1_one_block(msg: ptr<function, array<u32, 55>>, len: u32) -> Digest5 {
    // 64-byte message block in byte lanes (stored as u32 values 0..255).
    var block: array<u32, 64>;
    var i: u32 = 0u;
    loop {
        if (i >= 64u) {
            break;
        }
        // Zero initialize full block.
        block[i] = 0u;
        i = i + 1u;
    }

    // Copy candidate bytes into the block head.
    i = 0u;
    loop {
        if (i >= len) {
            break;
        }
        block[i] = (*msg)[i];
        i = i + 1u;
    }

    // SHA1 padding delimiter byte.
    block[len] = 0x80u;

    // SHA1 appends 64-bit big-endian message bit length at bytes 56..63.
    let bit_len = u64(len) << 3u;
    i = 0u;
    loop {
        if (i >= 8u) {
            break;
        }
        let shift = 56u - (8u * i);
        block[56u + i] = u32((bit_len >> shift) & u64(0xffu));
        i = i + 1u;
    }

    // Message schedule ring buffer (16 words) matching optimized MSL approach.
    var w: array<u32, 16>;
    i = 0u;
    loop {
        if (i >= 16u) {
            break;
        }
        // Load first 16 words from block as big-endian words.
        w[i] = load_be_u32(&block, i * 4u);
        i = i + 1u;
    }

    // SHA1 IV constants.
    let h0 = 0x67452301u;
    let h1 = 0xEFCDAB89u;
    let h2 = 0x98BADCFEu;
    let h3 = 0x10325476u;
    let h4 = 0xC3D2E1F0u;

    // Working state words.
    var a = h0;
    var b = h1;
    var c = h2;
    var d = h3;
    var e = h4;

    // 80 SHA1 rounds.
    var t: u32 = 0u;
    loop {
        if (t >= 80u) {
            break;
        }
        var wt: u32;
        if (t < 16u) {
            // First 16 rounds read directly from schedule seed.
            wt = w[t];
        } else {
            // Ring-buffer schedule recurrence:
            // W[t] = rol1(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16])
            let s = t & 15u;
            wt = rotl(w[(s + 13u) & 15u] ^ w[(s + 8u) & 15u] ^ w[(s + 2u) & 15u] ^ w[s], 1u);
            w[s] = wt;
        }

        // Main compression step.
        let temp = rotl(a, 5u) + sha1_f(t, b, c, d) + e + sha1_k(t) + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
        t = t + 1u;
    }

    // Add working state back into IV.
    var out: Digest5;
    out.a = a + h0;
    out.b = b + h1;
    out.c = c + h2;
    out.d = d + h3;
    out.e = e + h4;
    return out;
}

// Try to claim first-match flag and store index if we win the race.
// Equivalent to MSL record_first().
fn record_first(idx: u64) {
    let won = atomicCompareExchangeWeak(&found_flag.value, 0u, 1u);
    if (won.exchanged) {
        found_index.value = idx;
    }
}

// Append match index in all-matches mode, capped by params.max_matches.
// Equivalent to MSL record_all().
fn record_all(idx: u64) {
    let pos = atomicAdd(&match_count.value, 1u);
    if (pos < params.max_matches) {
        match_indices.values[pos] = idx;
    }
}

// Main brute-force kernel entrypoint.
// Equivalent to MSL kernel sha1_brute_force().
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn sha1_brute_force(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    // Fast global early-out in first-match mode.
    if (params.mode == 0u && atomicLoad(&found_flag.value) != 0u) {
        return;
    }

    // Global thread index.
    let gid = u64(global_id.x);
    // Total participating threads across all workgroups.
    let grid_threads = u64(num_workgroups.x) * u64(WORKGROUP_SIZE);
    // Per-iteration stride: every thread skips ahead by one full grid worth of work.
    let step = grid_threads * u64(params.candidates_per_thread);
    // First candidate index assigned to this thread.
    let base = gid * u64(params.candidates_per_thread);

    // Per-thread candidate byte buffer.
    var msg: array<u32, 55>;

    // Outer striding loop over disjoint chunks in search space.
    var start = base;
    loop {
        if (start >= params.search_space) {
            break;
        }

        // Re-check first-match flag each outer iteration to reduce wasted work.
        if (params.mode == 0u && atomicLoad(&found_flag.value) != 0u) {
            return;
        }

        // Inner fixed upper bound matches host/kernel contract with candidates_per_thread <= 32.
        var k: u32 = 0u;
        loop {
            if (k >= params.candidates_per_thread || k >= 32u) {
                break;
            }

            // Candidate index for this lane in the thread-local chunk.
            let idx = start + u64(k);
            if (idx >= params.search_space) {
                break;
            }

            // Runtime alphabet branch. Same strategy as MSL for single-kernel charset selection.
            if (params.alphabet_id == 0u) {
                // MSL counterpart: map_lower().
                map_lower(idx, params.len, &msg);
            } else if (params.alphabet_id == 1u) {
                // MSL counterpart: map_lowernum().
                map_lowernum(idx, params.len, &msg);
            } else {
                // MSL counterpart: map_printable().
                map_printable(idx, params.len, &msg);
            }

            // Hash candidate and compare all 5 digest words.
            let digest = sha1_one_block(&msg, params.len);
            if (digest.a == params.target_a
                && digest.b == params.target_b
                && digest.c == params.target_c
                && digest.d == params.target_d
                && digest.e == params.target_e) {
                if (params.mode == 0u) {
                    // First mode: publish winner and stop this thread.
                    record_first(idx);
                    return;
                }
                // All mode: append and continue scanning.
                record_all(idx);
            }

            k = k + 1u;
        }

        // Advance to next grid-strided chunk.
        start = start + step;
    }
}
