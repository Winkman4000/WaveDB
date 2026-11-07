"""THE WIDENED TRIPLE KEY (2026-11-07): tripletop's packed key had fixed fields (kcol 25b | minute 6b | tcol 23b) and
took only EXTRACT(minute). Now each field is as wide as its codes need (bits(V_kcol), bits of the part's top value,
bits(V_tcol), at most 63 in all) and the part is any clock or calendar field with a small fixed range (second, minute,
hour, day, month, quarter, dow, isodow, doy, and the named forms MINUTE(x), HOUR(x), ...). Fields with no fixed range
(year, epoch, week) and a group key that is another part than the projection's are declined. On the sparse-default
fixture (tm the time staircase, s the sparse phrase), wide (w, near-unique), mid (u), narrow (c) and time (tm) persons,
both walks (the checklists and tt_survivors), the two-key form too. Every answer DuckDB's (ties at the LIMIT cut legal)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))
import controller
import test_sparse_default as F

PARTS = ('minute', 'second', 'hour', 'dow', 'isodow', 'day', 'month', 'quarter', 'doy')
PERSONS = ('u', 'w', 'c', 'tm')


def _q3(person, part, lim=10, form='extract(%s FROM tm)'):
    return ("SELECT %s, %s AS m, s, COUNT(*) AS cc FROM t GROUP BY %s, m, s ORDER BY cc DESC LIMIT %d"
            % (person, form % part, person, lim))


def _check(q, served=True):
    db, con = F._fixture()
    got = F._rows(db.run(q))
    assert F._legal(q, got, con), (q, controller._SERVED[0], got[:3])
    if served is not None:
        assert (controller._SERVED[0] == 'tripletop') == served, (q, controller._SERVED[0])
    return got


def test_parts_by_persons():
    """every fixed-range part, every person width, through the checklist walk"""
    for person in PERSONS:
        for part in PARTS:
            _check(_q3(person, part))


def test_old_walk_parts():
    """the same through tt_survivors (WDB_TT_CHECKLISTS=0's road): its shifts are the arguments too"""
    import wdb_tripletop as TT
    TT._TTBITS[0] = False
    try:
        for person in ('u', 'w', 'c'):
            for part in ('minute', 'second', 'hour', 'doy'):
                _check(_q3(person, part, lim=7))
    finally:
        TT._TTBITS[0] = True


def test_named_forms_and_group_by_expr():
    """MINUTE(tm) / HOUR(tm), and a GROUP BY that repeats the expression instead of the alias"""
    _check(_q3('u', '', form='minute(tm)%s'))
    _check(_q3('w', '', form='hour(tm)%s'))
    _check("SELECT u, extract(second FROM tm) AS m, s, COUNT(*) AS cc FROM t "
           "GROUP BY u, extract(second FROM tm), s ORDER BY cc DESC LIMIT 10")


def test_layout_from_the_data():
    """the fields' widths are the data's: the wide person's codes sit above a 6-bit minute above the phrase's bits"""
    import sqlglot, wdb_tripletop as TT
    db, _ = F._fixture()
    db.run('SELECT COUNT(*) FROM t')
    seg = next(ent[1] for ent in db._seg_cache.values() if 's' in ent[1].cols and 'tm' in ent[1].cols)   # t's own
    tb = (int(seg.cols['s']['V']) - 1).bit_length()
    sp = TT.detect(seg, sqlglot.parse_one(_q3('w', 'minute'), read='duckdb'), {})
    assert sp['msh'] == tb and sp['ksh'] == tb + 6 and sp['mmask'] == 63 and sp['nm'] == 60, sp
    sp = TT.detect(seg, sqlglot.parse_one(_q3('c', 'hour'), read='duckdb'), {})
    assert sp['msh'] == tb and sp['ksh'] == tb + 5 and sp['mmask'] == 31 and sp['nm'] == 24, sp
    sp = TT.detect(seg, sqlglot.parse_one(_q3('u', 'dow'), read='duckdb'), {})
    assert sp['ksh'] == tb + 3 and sp['nm'] == 7, sp


def test_two_key_form():
    for person in ('u', 'w', 'c', 'tm'):
        _check("SELECT %s, s, COUNT(*) AS cc FROM t GROUP BY %s, s ORDER BY cc DESC LIMIT 10" % (person, person))


def test_declined():
    """no fixed range, or two parts: another read answers, still DuckDB's"""
    for part in ('year', 'week', 'epoch'):
        _check(_q3('u', part), served=False)
    _check("SELECT u, extract(minute FROM tm) AS m, extract(hour FROM tm) AS h, s, COUNT(*) AS cc FROM t "
           "GROUP BY u, m, h, s ORDER BY cc DESC LIMIT 10", served=False)


if __name__ == '__main__':
    n = 0
    for nm, fn in sorted(globals().items()):
        if nm.startswith('test_') and callable(fn):
            fn(); n += 1; print('ok', nm)
    print('%d passed' % n)
