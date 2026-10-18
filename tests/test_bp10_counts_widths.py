"""enc 10's census at every width (2026-10-18): bp10_counts equals np.bincount of bp10_decode for code widths
1-16, with run blocks and bitpack blocks mixed, a partial last block, and values at both ends of the width.
The byte census (widths 1, 2, 4, 8), the 8-row words (3, 5, 6, 7) and the row walk (9-16, the last block) each
run here. Blocks are built in the dress's own layout: a directory of (offset << 1 | is_run), a run block as
u16 count of runs then (u16 length, u16 value) pairs, a bitpack block as rows x bits MSB-first."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import wdb_kernels as WK


def _pack(codes, bits):
    b = ((codes[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    return np.packbits(b).tobytes()


def _dress(codes, bits, run_every):
    B = 4096
    n = codes.size
    nb = (n + B - 1) // B
    pay = bytearray(); dirX = np.zeros(nb, np.int64)
    for k in range(nb):
        blk = codes[k * B:(k + 1) * B]
        if k % run_every == 0:
            starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
            ends = np.r_[starts[1:], blk.size]
            dirX[k] = (len(pay) << 1) | 1
            pay += np.uint16(starts.size).tobytes()
            for s, e in zip(starts, ends):
                pay += np.uint16(e - s).tobytes() + np.uint16(blk[s]).tobytes()
        else:
            dirX[k] = len(pay) << 1
            pay += _pack(blk, bits)
    return np.frombuffer(bytes(16) + bytes(pay) + bytes(16), np.uint8), dirX, 16


def test_bp10_counts_every_width():
    rng = np.random.default_rng(18)
    for bits in range(1, 17):
        V = 1 << bits
        n = 4096 * 37 + 1234                                           # a partial last block
        codes = rng.integers(0, V, n)
        codes[:5000] = np.repeat(rng.integers(0, V, 50), 100)           # runs for the run blocks
        codes[10] = V - 1; codes[11] = 0                                # both ends of the width
        if bits == 1:
            codes[4096 * 5:4096 * 6] = 1
        buf, dirX, pay = _dress(codes.astype(np.int64), bits, 4)
        dec = np.zeros(n, np.uint16 if bits > 8 else np.uint8)
        WK.bp10_decode(buf, dirX, pay, bits, n, dec)
        assert np.array_equal(dec, codes), bits
        for T in (1, 3, 16):
            got = np.asarray(WK.bp10_counts(buf, dirX, np.int64(pay), np.int64(bits), np.int64(n),
                                            np.int64(V + 2), np.int64(T)))
            assert np.array_equal(got, np.bincount(codes, minlength=V + 2)), (bits, T)


def test_enc5_counts_byte_census():
    rng = np.random.default_rng(19)
    for n in (1, 2, 1_000_001, 1_000_000):
        nib = rng.integers(0, 16, n)
        hot = rng.permutation(400)[:15]
        npatch = int((nib == 15).sum())
        patches = rng.integers(0, 400, npatch).astype(np.uint16)
        packed = np.zeros((n + 1) // 2, np.uint8)
        packed[:] = nib[0::2]
        packed[:n // 2] |= (nib[1::2] << 4).astype(np.uint8)
        ref = np.bincount(np.r_[hot[nib[nib < 15]], patches].astype(np.int64), minlength=402)
        for T in (1, 5, 16):
            got = np.asarray(WK.enc5_counts(packed, hot.astype(np.int64), patches, np.int64(n), np.int64(402),
                                            np.int64(T)))
            assert np.array_equal(got, ref), (n, T)
