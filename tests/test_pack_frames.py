"""THE SERIALIZE SPEEDUPS WRITE THE SAME BYTES (2026-10-01): the compiled bit-pack equals the numpy
bit-matrix pack it replaced (every width 1..56, every code dtype, negative codes, ragged tails, a
chunk edge), and the threaded frames equal one compressor going frame by frame -- the tag-3 and
tag-18 sections are byte-identical with the threads on and off."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import zstandard as zstd
import wdb_encode as E


def _old_pack(codes, bits):
    codes = np.asarray(codes); out = bytearray()
    step = max(8, (1 << 23) // max(1, bits)); step -= step % 8
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    for lo in range(0, codes.size, step):
        part = codes[lo:lo + step].astype(np.uint64)
        out += np.packbits(((part[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)).tobytes()
    return bytes(out)


def test_pack_every_width_and_dtype():
    rng = np.random.default_rng(11)
    for bits in range(1, 57):
        for dt in (np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64):
            if bits > np.dtype(dt).itemsize * 8:
                continue
            n = int(rng.integers(1, 3000))
            hi = min(bits, np.dtype(dt).itemsize * 8 - (1 if np.dtype(dt).kind == 'i' else 0))
            a = rng.integers(0, 1 << hi, n, dtype=np.uint64).astype(dt)
            assert E._pack_codes(a, bits) == _old_pack(a, bits), (bits, dt, n)


def test_pack_negative_bool_and_chunk_edge():
    rng = np.random.default_rng(12)
    a = rng.integers(-1000, 1000, 5001, dtype=np.int64)        # two's complement low bits, as astype(u64)
    for bits in (3, 11, 33, 56):
        assert E._pack_codes(a, bits) == _old_pack(a, bits), bits
    b = rng.integers(0, 2, 9999).astype(bool)
    assert E._pack_codes(b, 1) == _old_pack(b, 1)
    big = rng.integers(0, 2, (1 << 23) + 77, dtype=np.int64)   # crosses the old chunk edge
    assert E._pack_codes(big, 1) == _old_pack(big, 1)
    assert E._pack_codes(np.zeros(0, np.int64), 5) == b''


def test_frames_threaded_equal_the_loop():
    rng = np.random.default_rng(13)
    a = np.repeat(rng.integers(0, 5000, 40000), 50).astype(np.uint16)   # 2M rows with runs
    BR = 65536
    n = (a.size + BR - 1) // BR
    cz = zstd.ZstdCompressor(level=9)
    loop = [cz.compress(a[i:i + BR].tobytes()) for i in range(0, a.size, BR)]
    got = E._zframes(9, n, lambda k: a[k * BR:(k + 1) * BR].tobytes())
    assert got == loop


def test_code_section_same_bytes_threads_on_off():
    rng = np.random.default_rng(14)
    N = 3_000_000
    cases = [(np.repeat(rng.integers(0, 3000, N // 30), 30), 12),          # zstd / blocked frames
             (rng.integers(0, 1 << 20, N), 20),                              # wide: packed frames (tag 18)
             (np.where(rng.random(N) < 0.97, 0, rng.integers(1, 9, N)), 4)]  # sparse / tiered
    for codes, bits in cases:
        codes = codes.astype(np.uint32)
        old = E._FRAME_THREADS
        try:
            E._FRAME_THREADS = 1
            ref = E._code_section(codes, bits)
            E._FRAME_THREADS = 4
            got = E._code_section(codes, bits)
        finally:
            E._FRAME_THREADS = old
        assert got == ref, (bits, got[:1], ref[:1])
