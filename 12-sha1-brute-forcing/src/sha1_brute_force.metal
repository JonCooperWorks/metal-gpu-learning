#include <metal_stdlib>
using namespace metal;

struct KernelParams {
    uint len;
    uint radix;
    ulong search_space;
    uint candidates_per_thread;
    uint mode;          // 0=first, 1=all
    uint max_matches;
    uint alphabet_id;   // 0=lower, 1=lowernum, 2=printable
    uint target_a;
    uint target_b;
    uint target_c;
    uint target_d;
    uint target_e;
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
        out[i] = (uchar)(0x20u + digit);
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
    thread uchar block[64];
    for (uint i = 0; i < 64; i++) {
        block[i] = 0u;
    }

    for (uint i = 0; i < len; i++) {
        block[i] = msg[i];
    }
    block[len] = 0x80u;

    // SHA-1 length field is 64-bit big-endian bit length.
    ulong bit_len = (ulong)len * 8ul;
    for (uint i = 0; i < 8; i++) {
        block[56 + i] = (uchar)((bit_len >> (56ul - 8ul * (ulong)i)) & 0xfful);
    }

    thread uint w[80];
    for (uint t = 0; t < 16; t++) {
        w[t] = load_be_u32(&block[t * 4]);
    }
    for (uint t = 16; t < 80; t++) {
        w[t] = rotl(w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16], 1u);
    }

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

    for (uint t = 0; t < 80; t++) {
        uint temp = rotl(a, 5u) + sha1_f(t, b, c, d) + e + sha1_k(t) + w[t];
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

    for (ulong start = base; start < params.search_space; start += step) {
        if (params.mode == 0u && atomic_load_explicit(found_flag, memory_order_relaxed) != 0u) {
            return;
        }

        #pragma unroll
        for (uint k = 0; k < 32u; k++) {
            if (k >= params.candidates_per_thread) break;

            ulong idx = start + (ulong)k;
            if (idx >= params.search_space) break;

            if (params.alphabet_id == 0u) {
                map_lower(idx, params.len, msg);
            } else if (params.alphabet_id == 1u) {
                map_lowernum(idx, params.len, msg);
            } else {
                map_printable(idx, params.len, msg);
            }

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
        }
    }
}
