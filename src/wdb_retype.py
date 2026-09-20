"""wdb_retype.py -- re-type integer columns as temporal (Date/DateTime) in place.

A retype is a metadata + dict-format change, NOT a row reorder and NOT a value
change: a dt=0 int column storing days-since-epoch (or epoch seconds) already holds
exactly the datetime64 integer representation. Only the dict byte-encoding differs
(dt=0 stores ASCII decimal; dt=3 stores packed <q int64), so we re-encode just the
dict+codes for the targeted columns and BYTE-COPY every other column blob verbatim.
Codes are re-derived from the decoded values in stored row order, so all byte-copied
columns stay aligned. If the cluster key is retyped, the .cluster sidecar's dtype/aux
are patched (values stay int64 -- slice_for_predicate + _lit_for_col both compare ints).

CLI: python wdb_retype.py <in.wdb> <out.wdb> EventDate=D EventTime=s
  (RHS is the datetime64 unit: D=day, s=second, ms, us, ns, ...)
"""
import sys, struct, pickle, os, time
import numpy as np
import zstandard as zstd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wdb_engine import Segment
from wdb_encode import _prep_column, _serialize_column, _unit_code, ZSTD_LEVEL


def _column_spans(buf):
    """Replicate the WVDB4 parse walk, returning [(name, start, end)] blob byte-ranges.
    Mirrors Segment.__init__ exactly; asserts a clean partition of the file as a self-check."""
    assert bytes(buf[:5]) == b'WVDB4', "not a WVDB4 segment"
    off = 5
    n_cols = struct.unpack_from('<H', buf, off)[0]; off += 2
    N = struct.unpack_from('<I', buf, off)[0]; off += 4
    spans = []
    for _ in range(n_cols):
        start = off
        nl = struct.unpack_from('<H', buf, off)[0]; off += 2
        nm = bytes(buf[off:off+nl]).decode(); off += nl
        V = struct.unpack_from('<I', buf, off)[0]; off += 4
        bits = int(buf[off]); off += 1; dt = int(buf[off]); off += 1; mode = int(buf[off]); off += 1
        has_null = int(buf[off]); off += 1; aux = int(buf[off]); off += 1
        chunked = bool(aux & 0x40)
        n_dict = V - has_null
        if mode == 0:
            for _ in range(n_dict):
                vl = struct.unpack_from('<I', buf, off)[0]; off += 4 + vl
            # mode 0 has a per-row code section -> fall through (do NOT continue)
        elif mode == 2:
            zlen = struct.unpack_from('<I', buf, off)[0]; off += 4 + zlen
        elif mode == 3:
            off += 2; Vx = struct.unpack_from('<I', buf, off)[0]; off += 4
            for _ in range(n_dict):
                vl = struct.unpack_from('<I', buf, off)[0]; off += 4 + vl
            off += (Vx * bits + 7) // 8
            spans.append((nm, start, off)); continue
        elif mode == 4:
            n_exc = struct.unpack_from('<I', buf, off + 28)[0]
            blob_len = 32 if n_exc == 0 else 36 + struct.unpack_from('<I', buf, off + 32)[0]
            off += blob_len
            spans.append((nm, start, off)); continue
        elif mode == 5:
            zll = struct.unpack_from('<I', buf, off)[0]; off += 4 + zll
            zvl = struct.unpack_from('<I', buf, off)[0]; off += 4 + zvl
            spans.append((nm, start, off)); continue
        else:  # mode 1 front-coded (optionally chunked)
            off += 2  # R
            if chunked:
                off += 4  # CH
                nch = struct.unpack_from('<I', buf, off)[0]; off += 4
                nr = struct.unpack_from('<I', buf, off)[0]; off += 4
                off += nr * 4
                off += 4  # fclen
                off += nch * 4  # chunk_ustart
                czlen = np.asarray(buf[off:off + nch * 4]).view(np.uint32); off += nch * 4
                off += int(czlen.astype(np.int64).sum())
            else:
                nr = struct.unpack_from('<I', buf, off)[0]; off += 4
                off += nr * 4
                off += 4  # fclen
                zlen = struct.unpack_from('<I', buf, off)[0]; off += 4 + zlen
        # shared code section (modes 1 and 2 reach here)
        code_enc = int(buf[off]); off += 1
        if code_enc == 0:
            off += (N * bits + 7) // 8
        elif code_enc == 2:                          # staircase: gbits u8 + nsteps u32 + gap-packed
            gbits = int(buf[off]); off += 1
            nsteps = struct.unpack_from('<I', buf, off)[0]; off += 4
            off += (nsteps * gbits + 7) // 8
        elif code_enc in (3, 18):                    # blocked / packed frames: width|bits u8 + BR u32 + nfr u32 + offs + frames
            off += 1
            nfr = struct.unpack_from('<II', buf, off)[1]; off += 8
            boffs = np.asarray(buf[off:off + (nfr + 1) * 4]).view(np.uint32); off += (nfr + 1) * 4
            off += int(boffs[-1])
        elif code_enc == 17:                         # raw packed codes
            b17 = buf[off]; off += 1
            n17 = struct.unpack_from('<I', buf, off)[0]; off += 4
            off += (n17 * b17 + 7) // 8 + 2
        elif code_enc == 15:                         # clock dress: 5 offs tables + partner + payload
            off += 1 + 2
            nfr15 = struct.unpack_from('<II', buf, off)[1]; off += 8
            tot15 = 0
            for _p in range(5):
                o15 = np.asarray(buf[off:off + (nfr15 + 1) * 4]).view(np.uint32)
                tot15 += int(o15[-1]); off += (nfr15 + 1) * 4
            pnl15 = struct.unpack_from('<H', buf, off)[0]; off += 2 + pnl15
            off += tot15
        elif code_enc == 16:                         # clock stub
            pnl16 = struct.unpack_from('<H', buf, off)[0]; off += 2 + pnl16
        elif code_enc == 14:                         # field planes v2: ywidth+ybase+FR+nfr + 3 offs tables + frames
            off += 1 + 2
            nfr14 = struct.unpack_from('<II', buf, off)[1]; off += 8
            tot14 = 0
            for _p in range(3):
                o14 = np.asarray(buf[off:off + (nfr14 + 1) * 4]).view(np.uint32)
                tot14 += int(o14[-1]); off += (nfr14 + 1) * 4
            off += tot14
        else:
            off += 1  # cwidth
            czlen = struct.unpack_from('<I', buf, off)[0]; off += 4 + czlen
        spans.append((nm, start, off))
    assert spans[0][1] == 11, spans[0]
    assert spans[-1][2] == len(buf), (spans[-1][2], len(buf))
    for i in range(1, len(spans)):
        assert spans[i][1] == spans[i - 1][2], (i, spans[i-1], spans[i])
    return spans


def redress(seg_path, cols, out_path, verbose=True):
    """THE REDRESS: re-run the code-section ELECTION for `cols` (modes 0/1/2) on the stored
    codes and splice the winner in; dictionary, order and every other column are BYTE-COPIED
    (spans come from the engine's own parse -- meta['blob'] / meta['code_off'] -- never a
    mirror that drifts). A measurement tool: the same segment in a new dress, nothing else
    moved. Returns {col: (old_enc, new_enc, old_bytes, new_bytes)}."""
    from wdb_encode import _code_section
    seg = Segment(seg_path)
    for c in cols:
        assert 'code_off' in seg.cols[c], f"{c}: no code section (mode {seg.cols[c].get('mode')})"
    report = {}
    CH = 64 << 20
    with open(out_path, 'wb') as out:
        out.write(seg.buf[:11].tobytes())
        for nm in seg.order:
            c = seg.cols[nm]; start, end = c['blob']
            if nm in cols:
                cs = c['code_off']
                codes = np.ascontiguousarray(np.asarray(seg._raw_codes(nm)), dtype=np.int64)
                seg._codes.pop(nm, None)
                sec = _code_section(codes, int(c['bits']), nm=nm)
                out.write(seg.buf[start:cs].tobytes()); out.write(sec)
                report[nm] = (int(c.get('code_enc', 0)), int(sec[0]), end - cs, len(sec))
                if verbose:
                    print(f"  redressed {nm}: enc {c.get('code_enc', 0)} -> {sec[0]}  "
                          f"codes {(end - cs) / 1e6:.1f} -> {len(sec) / 1e6:.1f} MB", flush=True)
                del codes, sec
            else:
                for a in range(start, end, CH):          # byte-copy in windows (an 8 GB blob is not a bytes object)
                    out.write(seg.buf[a:min(end, a + CH)].tobytes())
    return report


def retype(seg_path, retypes, out_path, verbose=True):
    """retypes: {colname: unit} e.g. {'EventDate':'D','EventTime':'s'}."""
    seg = Segment(seg_path)
    spans = _column_spans(seg.buf)
    names = [s[0] for s in spans]
    for c in retypes:
        assert c in names, f"{c} not in segment ({names})"
        assert seg.cols[c]['has_null'] == 0, f"{c} has nulls; retype path assumes non-null"
    zc = zstd.ZstdCompressor(level=ZSTD_LEVEL)
    with open(out_path, 'wb') as out:
        out.write(seg.buf[:11].tobytes())
        for nm, start, end in spans:
            if nm in retypes:
                unit = retypes[nm]
                iv = np.ascontiguousarray(np.asarray(seg.values(nm)), dtype=np.int64)
                arr = iv.view(f'datetime64[{unit}]')
                p = _prep_column(nm, arr, allow_seq=False)
                blob, _ = _serialize_column(p, zc)
                out.write(blob)
                if verbose:
                    print(f"  retyped {nm}: dt=3[{unit}] mode={p['mode']} V={p['V']} "
                          f"blob={len(blob)} (was {end-start})", flush=True)
            else:
                out.write(seg.buf[start:end].tobytes())
    sc = seg_path + '.cluster'
    if os.path.exists(sc):
        with open(sc, 'rb') as f:
            cm = pickle.load(f)
        if cm.get('key') in retypes:
            cm['dtype'] = 3
            cm['aux'] = _unit_code(retypes[cm['key']])
        with open(out_path + '.cluster', 'wb') as f:
            pickle.dump(cm, f)
        if verbose:
            print(f"  sidecar: key={cm.get('key')} dtype->{cm.get('dtype')} "
                  f"aux->{cm.get('aux')}", flush=True)


if __name__ == '__main__':
    inp, outp = sys.argv[1], sys.argv[2]
    rt = {}
    for a in sys.argv[3:]:
        k, u = a.split('='); rt[k] = u
    t0 = time.time()
    retype(inp, rt, outp)
    print(f"done in {time.time()-t0:.1f}s -> {outp}", flush=True)
