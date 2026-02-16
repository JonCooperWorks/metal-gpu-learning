// =============================================================================
// LESSON 11 METAL KERNELS: GPU MD5 BRUTE FORCING (THROUGHPUT-FIRST, TEACHING VERSION)
// =============================================================================
//
// This file implements a GPU brute force search for a *single target MD5 hash*.
//
// Core idea:
//   - We enumerate candidate strings from an index space: 0..(radix^len - 1)
//   - Each GPU thread tests many candidate indices (grid-stride loop).
//   - For each candidate:
//        1) map index -> base-radix digits
//        2) build candidate bytes
//        3) run MD5 on a single 512-bit block
//        4) compare digest with target
//
// Why "single block" only?
//   MD5 operates on 512-bit blocks. If the message length <= 55 bytes, after
//   adding the 0x80 byte and the 64-bit length (in bits), everything fits in
//   one 64-byte block. That is the common case for password cracking demos,
//   and it keeps the GPU kernel very fast.
//
// We expose three kernels with compile-time alphabets:
//   - lowercase      : 'a'..'z'                      (radix 26)
//   - lower+digits   : 'a'..'z' + '0'..'9'           (radix 36)
//   - printable ASCII: bytes 0x20..0x7e inclusive     (radix 95)
//
// Compile-time alphabet means:
//   - no global-memory alphabet lookup
//   - fewer loads, more ALU, more consistent throughput
//
// Host contract (Rust):
//   buffer(0)  constant KernelParams
//   buffer(1)  device atomic_uint* found_flag
//   buffer(2)  device ulong*       found_index
//   buffer(3)  device atomic_uint* match_count
//   buffer(4)  device ulong*       match_indices (len = max_matches)
//
// Notes on atomics:
//   - "first" mode uses found_flag + found_index and early-exits work.
//   - "all" mode uses match_count + match_indices to append hits.
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Kernel parameter block (must match Rust #[repr(C)] layout)
// -----------------------------------------------------------------------------
struct KernelParams {
    // Candidate length for this dispatch pass.
    uint len;
    // Radix for this dispatch (26, 36, 95). Included for diagnostics.
    uint radix;

    // Total number of candidates in this pass (= radix^len).
    // We keep it 64-bit because radix^len grows quickly.
    ulong total;

    // How many candidates each thread attempts per loop iteration.
    // This is a throughput knob: higher means more work per thread,
    // fewer global loop iterations, and (often) better ALU utilization.
    uint candidates_per_thread;

    // Match mode:
    //   0 = first match only (stop as soon as any thread finds a hit)
    //   1 = collect all matches (up to max_matches)
    uint mode;

    // Maximum number of matches to record in "all" mode.
    uint max_matches;

    // Target digest in little-endian words (MD5 outputs A,B,C,D as 32-bit LE).
    uint target_a;
    uint target_b;
    uint target_c;
    uint target_d;
};

// -----------------------------------------------------------------------------
// MD5 implementation (single block)
// -----------------------------------------------------------------------------

// MD5 overview (practical)
// ------------------------
// MD5 is a Merkle-Damgard hash built from a 512-bit compression function.
//
// For each 512-bit block, MD5:
//   1) parses the block as 16 little-endian u32 words X[0..15]
//   2) runs 64 steps grouped into 4 rounds of 16 steps
//   3) updates a 128-bit state (A,B,C,D), each a u32
//
// The state starts from fixed constants (the MD5 IV) and after the 64 steps,
// the working values are added back into the state (\"feed-forward\").
//
// Single-block constraint in this lesson:
//   MD5 padding appends a 64-bit length at the end of the final block.
//   If message length <= 55 bytes, the padding fits in one 64-byte block.
//   If message length >= 56 bytes, you need multiple blocks (not implemented).
//
// Why single-block is ideal for brute force:
//   Password candidates are usually short; staying in one block keeps the GPU
//   kernel branch-light and very fast.
//
// MD5 uses four boolean functions and a left-rotate.
inline uint md5_rotl(uint x, uint s) {
    return (x << s) | (x >> (32u - s));
}
inline uint F(uint x, uint y, uint z) { return (x & y) | (~x & z); }
inline uint G(uint x, uint y, uint z) { return (x & z) | (y & ~z); }
inline uint H(uint x, uint y, uint z) { return x ^ y ^ z; }
inline uint I(uint x, uint y, uint z) { return y ^ (x | ~z); }

// Load 32-bit little-endian word from 4 bytes.
inline uint load_le_u32(const thread uchar *p) {
    return (uint)p[0]
        | ((uint)p[1] << 8)
        | ((uint)p[2] << 16)
        | ((uint)p[3] << 24);
}

// MD5 constants
// -------------
// MD5 uses two fixed tables:
//
// 1) T[0..63]
//    T[i] = floor(abs(sin(i+1)) * 2^32)
//    These are standard constants (\"nothing up my sleeve\").
//
// 2) S[0..63]
//    Per-step rotate amounts (also standard).
//
// Keeping them in Metal `constant` memory is ideal:
//   - read-only
//   - identical across all threads
//   - typically cached efficiently
constant uint MD5_T[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu,
    0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
    0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau,
    0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu,
    0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
    0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u,
    0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u,
    0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
    0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u,
};

// Per-round rotation schedule.
constant uint MD5_S[64] = {
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
};

// Compute MD5 digest for a message of length <= 55 bytes.
//
// Input:
//   msg[0..len) = candidate bytes (password)
//   len         = candidate length in bytes
//
// Output:
//   out_a..out_d = digest words A,B,C,D for this one-block MD5\n+//
// Endianness (critical):
//   - The 512-bit block is parsed into 16 u32 words in little-endian order.\n+//   - The MD5 digest is commonly presented as 16 bytes, which correspond to\n+//     the little-endian words A,B,C,D concatenated.\n+//   - Host must compare using the same convention (Rust parses digest bytes\n+//     into LE u32 words before passing them to GPU).\n inline void md5_one_block(
inline void md5_one_block(
    const thread uchar *msg,
    uint len,
    thread uint &out_a,
    thread uint &out_b,
    thread uint &out_c,
    thread uint &out_d
) {
    // 1) Build the 64-byte block in thread-local memory.
    // We store bytes, then reinterpret as 16 little-endian u32 words.
    thread uchar block[64];

    // Copy message bytes.
    for (uint i = 0; i < len; i++) {
        block[i] = msg[i];
    }

    // Padding step 1: append a single '1' bit.
    //
    // Because our message is byte-aligned, we can append that '1' bit as the
    // byte 0x80 = 1000_0000, and the remaining bits of that byte are zeros.
    block[len] = 0x80u;

    // Padding step 2: write zeros until byte index 56.
    //
    // Bytes 56..63 store the message length in *bits* as a u64 (little-endian).
    // This is why this lesson requires len <= 55 for the one-block path.
    for (uint i = len + 1; i < 56; i++) {
        block[i] = 0u;
    }

    // Padding step 3: append the original message length in bits, as LE u64.
    ulong bit_len = (ulong)len * 8ul;
    for (uint i = 0; i < 8; i++) {
        block[56 + i] = (uchar)((bit_len >> (8ul * (ulong)i)) & 0xfful);
    }

    // 2) Initialize MD5 state (the MD5 IV).
    uint a = 0x67452301u;
    uint b = 0xefcdab89u;
    uint c = 0x98badcfeu;
    uint d = 0x10325476u;

    // 3) Load X[0..15] as little-endian u32 words.
    //
    // MD5's message schedule is just these 16 words; MD5 does not expand to\n+    // 64 words like SHA-256.
    thread uint X[16];
    for (uint i = 0; i < 16; i++) {
        X[i] = load_le_u32(&block[i * 4u]);
    }

    // 4) Main loop (64 steps).
    //
    // MD5 has 4 rounds of 16 steps. Each round uses:
    // - a different boolean function (F/G/H/I)
    // - a different formula for g, which selects the message word X[g]
    // - a different pattern of rotate amounts (still provided by MD5_S[i])
    //
    // Textbook step (variables named A,B,C,D):
    //   A = B + ROTL(A + f(B,C,D) + X[g] + T[i], S[i])
    //   (A,B,C,D) = (D,A,B,C)
    uint aa = a, bb = b, cc = c, dd = d;

    for (uint i = 0; i < 64; i++) {
        uint f;
        uint g;
        if (i < 16) {
            // Round 1 (i = 0..15):
            // - f = F(B,C,D)
            // - g = i
            f = F(bb, cc, dd);
            g = i;
        } else if (i < 32) {
            // Round 2 (i = 16..31):
            // - f = G(B,C,D)
            // - g = (5*i + 1) mod 16
            f = G(bb, cc, dd);
            g = (5u * i + 1u) & 15u;
        } else if (i < 48) {
            // Round 3 (i = 32..47):
            // - f = H(B,C,D)
            // - g = (3*i + 5) mod 16
            f = H(bb, cc, dd);
            g = (3u * i + 5u) & 15u;
        } else {
            // Round 4 (i = 48..63):
            // - f = I(B,C,D)
            // - g = (7*i) mod 16
            f = I(bb, cc, dd);
            g = (7u * i) & 15u;
        }

        // One MD5 step:
        // - compute (aa + f + T[i] + X[g])
        // - rotate-left by S[i]
        // - add into bb
        //
        // The tmp/aa/bb/cc/dd shuffles implement the register rotation:
        //   (A,B,C,D) = (D,A,B,C)
        uint tmp = dd;
        dd = cc;
        cc = bb;
        bb = bb + md5_rotl(aa + f + MD5_T[i] + X[g], MD5_S[i]);
        aa = tmp;
    }

    // 5) Feed-forward add.
    //
    // This is what makes MD5 iterative across multiple blocks:
    // each block's result is added to the incoming state.
    //
    // For a one-block message, this produces the final digest state.
    a += aa;
    b += bb;
    c += cc;
    d += dd;

    out_a = a;
    out_b = b;
    out_c = c;
    out_d = d;
}

// -----------------------------------------------------------------------------
// Index -> candidate byte mapping
// -----------------------------------------------------------------------------

// Each charset has its own mapping function.
// We build `len` bytes in `out[0..len)`.
//
// The mapping is little-endian in *digit space*:
//   idx = d0 + d1*radix + d2*radix^2 + ...
// where each digit chooses a character.
// This produces a deterministic enumeration order.

inline void map_lower(ulong idx, uint len, thread uchar *out) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 26ul);
        idx /= 26ul;
        out[i] = (uchar)('a' + digit);
    }
}

inline void map_lowernum(ulong idx, uint len, thread uchar *out) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 36ul);
        idx /= 36ul;
        // 0..25 => a..z, 26..35 => 0..9
        if (digit < 26u) {
            out[i] = (uchar)('a' + digit);
        } else {
            out[i] = (uchar)('0' + (digit - 26u));
        }
    }
}

inline void map_printable(ulong idx, uint len, thread uchar *out) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 95ul);
        idx /= 95ul;
        out[i] = (uchar)(0x20u + digit); // ' ' .. '~'
    }
}

// -----------------------------------------------------------------------------
// Match recording helpers
// -----------------------------------------------------------------------------

// "first" mode: only the first thread to claim sets found_index.
inline void record_first(
    device atomic_uint *found_flag,
    device ulong *found_index,
    ulong idx
) {
    // Compare-and-swap from 0 -> 1. If we win, write index.
    uint expected = 0u;
    bool won = atomic_compare_exchange_weak_explicit(
        found_flag,
        &expected,
        1u,
        memory_order_relaxed,
        memory_order_relaxed
    );
    if (won) {
        *found_index = idx;
    }
}

// "all" mode: append to match_indices up to max_matches.
inline void record_all(
    device atomic_uint *match_count,
    device ulong *match_indices,
    uint max_matches,
    ulong idx
) {
    uint pos = atomic_fetch_add_explicit(match_count, 1u, memory_order_relaxed);
    if (pos < max_matches) {
        match_indices[pos] = idx;
    }
}

// -----------------------------------------------------------------------------
// Shared brute force loop template
// -----------------------------------------------------------------------------

// All kernels share the same control flow; only the mapping differs.
// We implement three kernels separately so the compiler can inline mapping.

// Teaching note on grid-stride loops:
//   We want to be independent of exactly how many threads the host launches.
//   Each thread processes indices:
//      base = gid * candidates_per_thread
//      then base += grid_threads * candidates_per_thread
//   This keeps memory traffic low (we mostly do ALU) and evenly distributes work.

// Lowercase kernel.
kernel void md5_bruteforce_lower(
    constant KernelParams &params         [[ buffer(0) ]],
    device atomic_uint *found_flag        [[ buffer(1) ]],
    device ulong *found_index             [[ buffer(2) ]],
    device atomic_uint *match_count       [[ buffer(3) ]],
    device ulong *match_indices           [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]],
    uint grid_threads                     [[ threads_per_grid ]]
) {
    // Early exit if another thread already found a match in "first" mode.
    // This check is intentionally outside the inner loop.
    // In "all" mode we must *not* early-exit.
    if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
        return;
    }

    ulong step = (ulong)grid_threads * (ulong)params.candidates_per_thread;
    ulong base = (ulong)gid * (ulong)params.candidates_per_thread;

    thread uchar msg[55];

    for (ulong start = base; start < params.total; start += step) {
        if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
            return;
        }

        // Test a small batch per thread per outer iteration.
        // This reduces loop overhead and increases instruction-level parallelism.
        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            // We can't unroll to an arbitrary runtime candidates_per_thread.
            // Strategy: unroll a fixed max (32) and guard with `k < candidates_per_thread`.
            // Host defaults candidates_per_thread=8.
            if (k >= params.candidates_per_thread) break;

            ulong idx = start + (ulong)k;
            if (idx >= params.total) break;

            map_lower(idx, params.len, msg);

            uint a, b, c, d;
            md5_one_block(msg, params.len, a, b, c, d);

            if (a == params.target_a && b == params.target_b && c == params.target_c && d == params.target_d) {
                if (params.mode == 0u) {
                    record_first(found_flag, found_index, idx);
                    return;
                } else {
                    record_all(match_count, match_indices, params.max_matches, idx);
                }
            }
        }
    }
}

// Lowercase + digits kernel.
kernel void md5_bruteforce_lowernum(
    constant KernelParams &params         [[ buffer(0) ]],
    device atomic_uint *found_flag        [[ buffer(1) ]],
    device ulong *found_index             [[ buffer(2) ]],
    device atomic_uint *match_count       [[ buffer(3) ]],
    device ulong *match_indices           [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]],
    uint grid_threads                     [[ threads_per_grid ]]
) {
    if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
        return;
    }

    ulong step = (ulong)grid_threads * (ulong)params.candidates_per_thread;
    ulong base = (ulong)gid * (ulong)params.candidates_per_thread;

    thread uchar msg[55];

    for (ulong start = base; start < params.total; start += step) {
        if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
            return;
        }

        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            if (k >= params.candidates_per_thread) break;

            ulong idx = start + (ulong)k;
            if (idx >= params.total) break;

            map_lowernum(idx, params.len, msg);

            uint a, b, c, d;
            md5_one_block(msg, params.len, a, b, c, d);

            if (a == params.target_a && b == params.target_b && c == params.target_c && d == params.target_d) {
                if (params.mode == 0u) {
                    record_first(found_flag, found_index, idx);
                    return;
                } else {
                    record_all(match_count, match_indices, params.max_matches, idx);
                }
            }
        }
    }
}

// Printable ASCII kernel.
kernel void md5_bruteforce_printable(
    constant KernelParams &params         [[ buffer(0) ]],
    device atomic_uint *found_flag        [[ buffer(1) ]],
    device ulong *found_index             [[ buffer(2) ]],
    device atomic_uint *match_count       [[ buffer(3) ]],
    device ulong *match_indices           [[ buffer(4) ]],
    uint gid                              [[ thread_position_in_grid ]],
    uint grid_threads                     [[ threads_per_grid ]]
) {
    if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
        return;
    }

    ulong step = (ulong)grid_threads * (ulong)params.candidates_per_thread;
    ulong base = (ulong)gid * (ulong)params.candidates_per_thread;

    thread uchar msg[55];

    for (ulong start = base; start < params.total; start += step) {
        if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
            return;
        }

        #pragma unroll
        for (uint k = 0; k < 32; k++) {
            if (k >= params.candidates_per_thread) break;

            ulong idx = start + (ulong)k;
            if (idx >= params.total) break;

            map_printable(idx, params.len, msg);

            uint a, b, c, d;
            md5_one_block(msg, params.len, a, b, c, d);

            if (a == params.target_a && b == params.target_b && c == params.target_c && d == params.target_d) {
                if (params.mode == 0u) {
                    record_first(found_flag, found_index, idx);
                    return;
                } else {
                    record_all(match_count, match_indices, params.max_matches, idx);
                }
            }
        }
    }
}
