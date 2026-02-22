#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    // Candidate length for this dispatch pass.
    uint len;
    // Encoded alphabet enum from the host:
    //   0 = lower, 1 = lowernum, 2 = printable
    uint alphabet;

    // Total number of candidates in this pass (= radix^len).
    // We keep it 64-bit because radix^len grows quickly.
    ulong search_space;

    // How many candidates each thread attempts per loop iteration.
    // This is a throughput knob: higher means more work per thread,
    // fewer global loop iterations, and (often) better ALU utilization.
    uint candidates_per_thread;

    // Match mode:
    //   0 = first match only (stop as soon as any thread finds a hit)
    //   1 = collect all matches (up to max_matches)
    uint mode;          // 0=first, 1=all

    // Maximum number of matches to record in "all" mode.
    uint max_matches;

    // The target SHA-1 digest to match.
    // This is the hash that we are trying to find.
    // SHA-1 digest is 5 words (A,B,C,D,E) in little-endian order.
    uint target_a;
    uint target_b;
    uint target_c;    // 0=lower, 1=lowernum, 2=printable
    uint target_d;
    uint target_e;    // 0=lower, 1=lowernum, 2=printable
};

inline uint rotl(uint x, uint s) {
    return (x << s) | (x >> (32u - s));
}

inline uint load_be_u32(const thread uchar *p) {
    return ((uint)p[0] << 24)
        | ((uint)p[1] << 16)
        | ((uint)p[2] << 8)
        | (uint)p[3];
}

inline uint sha1_f(uint t, uint b, uint c, uint d) {
    if (t < 20u) return (b & c) | ((~b) & d);
    if (t < 40u) return b ^ c ^ d;
    if (t < 60u) return (b & c) | (b & d) | (c & d);
    return b ^ c ^ d;
}

inline uint sha1_k(uint t) {
    if (t < 20u) return 0x5A827999u;
    if (t < 40u) return 0x6ED9EBA1u;
    if (t < 60u) return 0x8F1BBCDCu;
    return 0xCA62C1D6u;
}

// Map a numeric candidate index to:
//   - `digits[i]`: the i-th base-26 digit (least-significant digit first)
//   - `msg[i]`: the corresponding ASCII byte
//
// We keep both because `digits` is cheap to increment, and `msg` is what SHA-1 consumes.
inline void map_lower_state(ulong idx, uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 26ul);
        idx /= 26ul;
        digits[i] = (uchar)digit;
        msg[i] = (uchar)('a' + digit);
    }
}

// Same idea as `map_lower_state`, but for [a-z0-9] (radix 36).
inline void map_lowernum_state(ulong idx, uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 36ul);
        idx /= 36ul;
        digits[i] = (uchar)digit;
        if (digit < 26u) {
            msg[i] = (uchar)('a' + digit);
        } else {
            msg[i] = (uchar)('0' + (digit - 26u));
        }
    }
}

// Same idea again, but for printable ASCII (0x20..0x7e, radix 95).
inline void map_printable_state(ulong idx, uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uint digit = (uint)(idx % 95ul);
        idx /= 95ul;
        digits[i] = (uchar)digit;
        msg[i] = (uchar)(0x20u + digit);
    }
}

// Odometer increment for radix-26.
// `digits[0]` is the least-significant "wheel", so it rolls every candidate.
// When a wheel wraps, we carry into the next wheel.
inline void increment_lower(uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uchar next = (uchar)(digits[i] + 1u);
        if (next < (uchar)26u) {
            digits[i] = next;
            msg[i] = (uchar)('a' + next);
            return;
        }
        digits[i] = 0u;
        msg[i] = (uchar)'a';
    }
}

// Odometer increment for radix-36 ([a-z0-9]).
inline void increment_lowernum(uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uchar next = (uchar)(digits[i] + 1u);
        if (next < (uchar)36u) {
            digits[i] = next;
            if (next < (uchar)26u) {
                msg[i] = (uchar)('a' + next);
            } else {
                msg[i] = (uchar)('0' + (next - (uchar)26u));
            }
            return;
        }
        digits[i] = 0u;
        msg[i] = (uchar)'a';
    }
}

// Odometer increment for radix-95 (printable ASCII).
inline void increment_printable(uint len, thread uchar *msg, thread uchar *digits) {
    for (uint i = 0; i < len; i++) {
        uchar next = (uchar)(digits[i] + 1u);
        if (next < (uchar)95u) {
            digits[i] = next;
            msg[i] = (uchar)(0x20u + next);
            return;
        }
        digits[i] = 0u;
        msg[i] = (uchar)0x20u;
    }
}

inline void sha1_one_block(
    const thread uchar *msg,
    uint len,
    thread uint &a,
    thread uint &b,
    thread uint &c,
    thread uint &d,
    thread uint &e
) {

    // We will build the block in thread-local memory.
    thread uchar block[64];
    for (uint i = 0; i < 64; i++) {
        block[i] = 0u;
    }

    // Let's copy the message bytes to the block.
    for (uint i = 0; i < len; i++) {
        block[i] = msg[i];
    }

    // Time for padding!
    block[len] = 0x80u;

    // SHA-1 length field is 64-bit big-endian bit length.
    ulong bit_len = (ulong)len * 8ul;
    for (uint i = 0; i < 8; i++) {
        block[56 + i] = (uchar)((bit_len >> (56ul - 8ul * (ulong)i)) & 0xfful);
    }


    // I had a uint[80] before but ChatGPT told me I could use a ring buffer yeah that thing from second year.
    // Holy shit how much weed did i smoke between uni and now??
    // thread uint w[80];
    // for (uint t = 0; t < 16; t++) {
    //     w[t] = load_be_u32(&block[t * 4]);
    // }
    // for (uint t = 16; t < 80; t++) {
    //     w[t] = rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1u);
    // }

    // So now we make a ring buffer of 16 elements.
    // Since the SHA1 hash is a 5 word digest, we only need 16 elements and we can do
    // some black magic to make it work much faster.
    thread uint w[16];

    // Now we can do the black magic.
    for (uint t = 0; t < 16; t++) {
        w[t] = load_be_u32(&block[t * 4u]);
    }

    // SHA-1 initial hash values.
    // We use them because the spec said so - I leave that to smarter people than me.
    // But one day I will know.
    uint h0 = 0x67452301u;
    uint h1 = 0xEFCDAB89u;
    uint h2 = 0x98BADCFEu;
    uint h3 = 0x10325476u;
    uint h4 = 0xC3D2E1F0u;

    a = h0;
    b = h1;
    c = h2;
    d = h3;
    e = h4;

    // More black magic fuckery.
    // The spec said so and I believe them.
    for (uint t = 0; t < 80; t++) {
        uint wt;
        if (t < 16u) {
            wt = w[t];
        } else {
            uint s = t & 15u;
            // Ring buffer voodoo annotation:
            //   s            == t mod 16       (slot for W[t] / old W[t-16])
            //   (s + 13) & 15 == (t - 3) mod 16
            //   (s + 8)  & 15 == (t - 8) mod 16
            //   (s + 2)  & 15 == (t - 14) mod 16
            // SHA-1 schedule: W[t] = rol1(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16]).
            wt = rotl(w[(s + 13u) & 15u] ^ w[(s + 8u) & 15u] ^ w[(s + 2u) & 15u] ^ w[s], 1u);
            w[s] = wt;
        }

        uint temp = rotl(a, 5u) + sha1_f(t, b, c, d) + e + sha1_k(t) + wt;
        e = d;
        d = c;
        c = rotl(b, 30u);
        b = a;
        a = temp;
    }

    a += h0;
    b += h1;
    c += h2;
    d += h3;
    e += h4;
}

inline void record_first(device atomic_uint *found_flag, device ulong *found_index, ulong idx) {
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

kernel void sha1_brute_force(
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
    // We keep base-radix digits alongside `msg` so the inner loop can do an odometer-style
    // increment instead of re-running `%` and `/` for every single candidate.
    //
    // Old way (slower inside the hot loop):
    //   - Every candidate did a fresh integer->base-radix conversion.
    //   - That means repeated `% radix` and `/ radix` in the inner loop.
    //
    // New way:
    //   - Convert `start` once per mini-batch (`map_*_state`).
    //   - For k=1..N, mutate the existing candidate in-place (`increment_*`).
    //
    // This tends to help when `candidates_per_thread > 1`, because we amortize the expensive
    // divide/mod work across the whole mini-batch.
    thread uchar digits[55];

    for (ulong start = base; start < params.search_space; start += step) {
        if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
            return;
        }

        // This thread works on a contiguous mini-batch: `start .. start + batch_count`.
        // We map the first candidate once, then increment in-place for the rest.
        //
        // Old code looked like this (inside the `k` loop):
        //     ulong idx = start + (ulong)k;
        //     if (params.alphabet == 0u) {
        //         map_lower(idx, params.len, msg);
        //     } else if (params.alphabet == 1u) {
        //         map_lowernum(idx, params.len, msg);
        //     } else {
        //         map_printable(idx, params.len, msg);
        //     }
        //
        // Which is perfectly correct, but it remaps from scratch every time.
        //
        // The new code below keeps `msg` and `digits` hot in thread-local memory and does
        // a carry-propagating increment between candidates (like adding 1 to a number).
        ulong remaining = params.search_space - start;
        uint batch_count = params.candidates_per_thread;
        if ((ulong)batch_count > remaining) {
            batch_count = (uint)remaining;
        }
        if (batch_count == 0u) {
            break;
        }

        // Determine the appropriate mapping function based on the alphabet enum.
        // Same story as before: the params are uniform, so every thread takes the same path.
        //
        // Important detail:
        //   `start` is the FIRST candidate in this mini-batch, so this is the only place where
        //   we pay the full base conversion cost. Later candidates are derived by incrementing.
        if (params.alphabet == 0u) {
            map_lower_state(start, params.len, msg, digits);
        } else if (params.alphabet == 1u) {
            map_lowernum_state(start, params.len, msg, digits);
        } else {
            map_printable_state(start, params.len, msg, digits);
        }

        #pragma unroll
        for (uint k = 0; k < 32u; k++) {
            if (k >= batch_count) break;

            ulong idx = start + (ulong)k;

            uint a, b, c, d, e;
            sha1_one_block(msg, params.len, a, b, c, d, e);

            if (a == params.target_a &&
                b == params.target_b &&
                c == params.target_c &&
                d == params.target_d &&
                e == params.target_e) {
                if (params.mode == 0u) {
                    record_first(found_flag, found_index, idx);
                    return;
                }
                record_all(match_count, match_indices, params.max_matches, idx);
            }

            // Advance to the next candidate in this thread's contiguous mini-batch.
            // This is the whole point of the optimization: carry propagation is much cheaper
            // than converting `idx` -> base-radix digits from scratch every time.
            //
            // Example (lower, len=3):
            //   "aaz" is stored as digits [25,0,0] because digit[0] is least-significant.
            //   increment -> wrap digit[0] to 0 ('a'), carry into digit[1]
            //   result is "aba" (digits [0,1,0]).
            if (k + 1u < batch_count) {
                if (params.alphabet == 0u) {
                    increment_lower(params.len, msg, digits);
                } else if (params.alphabet == 1u) {
                    increment_lowernum(params.len, msg, digits);
                } else {
                    increment_printable(params.len, msg, digits);
                }
            }
        }
    }
}
