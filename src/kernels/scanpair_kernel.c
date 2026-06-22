/* scanpair_kernel.c -- AVX-512 two-stage filtered 2-key COUNT(*) top-K helper for wdb_scanpair.
 *
 * The high-match regime of a 2-key COUNT(*) top-K with an equality filter on a non-key column C
 * is bounded by the AGGREGATION (counting the (a,b) pairs of the surviving rows), not the scan.
 * This kernel reads fewer bytes and counts instead of sorting:
 *   Stage 1 (scan, AVX-512): scan ONLY the filter column C -- in its NATIVE width (u8/u16/u32) --
 *            and compress-store the surviving row positions. A and B are never read for the rows
 *            the filter discards.
 *   Stage 2 (gather+fuse, scalar): gather A and B at just those positions (native width, no
 *            full-column upcast) and fuse each pair into one key a*Vb + b.
 *   Stage 3 (tally, scalar hash): count distinct fused keys with an open-addressing hash table
 *            sized to the pairs that OCCUR (not the full a*b key space) -- counts, never sorts.
 *
 * Built lazily at runtime by wdb_kernel.py with gcc -march=native; used only when AVX-512 is
 * present, otherwise the engine falls back to the numpy path with identical results.
 *
 * NATIVE WIDTH is the whole point: WaveDB stores codes as uint8/uint16/uint32 by cardinality.
 * Requiring uint32 would force a 100M-element upcast per call that erases the win. So stage 1 has
 * one scan variant per width, and stage 2 reads a/b through width-correct pointers.
 */
#include <immintrin.h>
#include <stdint.h>

/* ---- Stage 1: scan C alone, native width, compress surviving row indices ---- */

/* uint32 codes: 16 lanes per AVX-512 vector */
int64_t wdb_scan_u32(const uint32_t* cC, int64_t N, uint32_t vC, int32_t* out_pos) {
    int64_t m = 0;
    __m512i vc = _mm512_set1_epi32((int)vC);
    __m512i iota = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    int64_t i = 0;
    for (; i + 16 <= N; i += 16) {
        __m512i c = _mm512_loadu_si512((const void*)(cC + i));
        __mmask16 keep = _mm512_cmpeq_epu32_mask(c, vc);
        if (!keep) continue;
        __m512i idx = _mm512_add_epi32(_mm512_set1_epi32((int)i), iota);
        _mm512_mask_compressstoreu_epi32(out_pos + m, keep, idx);
        m += __builtin_popcount(keep);
    }
    for (; i < N; ++i) if (cC[i] == vC) out_pos[m++] = (int32_t)i;
    return m;
}

/* uint16 codes: compare 32 lanes per vector (AVX-512BW), emit surviving indices from the mask.
 * The vector compare over all N is the bandwidth-bound work; emitting the few survivor indices
 * per 32-lane block via the mask bits is cheap. */
int64_t wdb_scan_u16(const uint16_t* cC, int64_t N, uint16_t vC, int32_t* out_pos) {
    int64_t m = 0;
    __m512i vc = _mm512_set1_epi16((short)vC);
    int64_t i = 0;
    for (; i + 32 <= N; i += 32) {
        __m512i c = _mm512_loadu_si512((const void*)(cC + i));
        __mmask32 keep = _mm512_cmpeq_epu16_mask(c, vc);
        while (keep) {
            int b = __builtin_ctz(keep);          /* index of lowest set bit */
            out_pos[m++] = (int32_t)(i + b);
            keep &= keep - 1;                      /* clear lowest set bit */
        }
    }
    for (; i < N; ++i) if (cC[i] == vC) out_pos[m++] = (int32_t)i;
    return m;
}

/* uint8 codes: compare 64 lanes per vector (AVX-512BW), emit surviving indices from the mask. */
int64_t wdb_scan_u8(const uint8_t* cC, int64_t N, uint8_t vC, int32_t* out_pos) {
    int64_t m = 0;
    __m512i vc = _mm512_set1_epi8((char)vC);
    int64_t i = 0;
    for (; i + 64 <= N; i += 64) {
        __m512i c = _mm512_loadu_si512((const void*)(cC + i));
        __mmask64 keep = _mm512_cmpeq_epu8_mask(c, vc);
        while (keep) {
            int b = __builtin_ctzll(keep);
            out_pos[m++] = (int32_t)(i + b);
            keep &= keep - 1;
        }
    }
    for (; i < N; ++i) if (cC[i] == vC) out_pos[m++] = (int32_t)i;
    return m;
}

/* ---- Stage 2: gather A,B at survivor positions, native width, fuse a*Vb+b into int64 keys ----
 * Scalar: runs only over the m survivors (not N), and is memory-latency bound (gathering at
 * scattered positions), so vectorizing buys little; correctness + native-width reads matter.
 * aw/bw are the byte widths (1,2,4) of the A and B code arrays. We read each through the correct
 * pointer type -- NO full-column upcast. Keys are int64 (the fused value a*Vb+b always fits;
 * tally memory is bounded by distinct pairs, not key width). */
#define RDA(p,i) (aw==1 ? (int64_t)((const uint8_t*)(p))[i] : aw==2 ? (int64_t)((const uint16_t*)(p))[i] : (int64_t)((const uint32_t*)(p))[i])
#define RDB(p,i) (bw==1 ? (int64_t)((const uint8_t*)(p))[i] : bw==2 ? (int64_t)((const uint16_t*)(p))[i] : (int64_t)((const uint32_t*)(p))[i])

void wdb_gather_fuse(const void* cA, int aw, const void* cB, int bw,
                     const int32_t* pos, int64_t m, int64_t Vb, int64_t* out_keys) {
    for (int64_t i = 0; i < m; ++i) {
        int32_t p = pos[i];
        int64_t av = RDA(cA, p);
        int64_t bv = RDB(cB, p);
        out_keys[i] = av * Vb + bv;
    }
}

/* ---- Stage 3: open-addressing hash tally over the fused keys ----
 * Counts each distinct key. Table sized to 1<<capbits (caller picks ~2x distinct, power of two).
 * htab_k/htab_c are scratch (size 1<<capbits); out_k/out_c receive the distinct (key,count).
 * Returns the number of distinct keys. Multiplicative hash, linear probe, -1 = empty slot. */
int64_t wdb_tally(const int64_t* keys, int64_t m, int capbits,
                  int64_t* htab_k, int64_t* htab_c, int64_t* out_k, int64_t* out_c) {
    int64_t cap = 1LL << capbits, mask = cap - 1;
    /* Fill ceiling: never let the table get so full that linear probing degenerates / can't insert.
     * If distinct keys would exceed this, bail with -1 so the caller falls back to the numpy path
     * (this also makes a too-small capped table SAFE rather than an infinite probe loop). */
    int64_t ceiling = cap - (cap >> 3);          /* 87.5% load max */
    int64_t distinct = 0;
    for (int64_t i = 0; i < cap; ++i) htab_k[i] = -1;
    for (int64_t i = 0; i < m; ++i) {
        int64_t k = keys[i];
        int64_t h = (k * 2654435761LL) & mask;
        while (1) {
            if (htab_k[h] == -1) {
                if (distinct >= ceiling) return -1;   /* table too full -> signal fallback */
                htab_k[h] = k; htab_c[h] = 1; distinct++; break;
            }
            else if (htab_k[h] == k) { htab_c[h]++; break; }
            else h = (h + 1) & mask;
        }
    }
    int64_t d = 0;
    for (int64_t h = 0; h < cap; ++h)
        if (htab_k[h] != -1) { out_k[d] = htab_k[h]; out_c[d] = htab_c[h]; d++; }
    return d;
}
