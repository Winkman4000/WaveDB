"""Unit tests for wdb_policies -- each decision guard ("policy") in isolation.

This is the anti-cheat layer for the policy refactor: a query producing the right answer end-to-end
does NOT prove the guards are correct (it could pass via another path). These tests pin each policy's
pass/fail behavior directly, with tiny hand-built inputs, so a policy can't silently rot. The matching
*decline* tests in test_groupdistinct (operator must NOT fire on out-of-scope shapes) are the second
layer: together they prove the guards both admit and reject for the right reasons."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import sqlglot
import wdb_policies as P

def _t(sql):
    return sqlglot.parse_one(sql, read='duckdb')

class _FakeSeg:
    """Minimal stand-in: policies only read .cols and .presence_mask()."""
    def __init__(self, cols, presence=None):
        self.cols = cols
        self._presence = presence
    def presence_mask(self):
        return self._presence


# --- query-shape guards ---

def test_no_joins():
    assert P.no_joins(_t("SELECT a FROM t")) is True
    assert P.no_joins(_t("SELECT a FROM t JOIN u ON t.id = u.id")) is False

def test_no_select_distinct():
    assert P.no_select_distinct(_t("SELECT a FROM t")) is True
    assert P.no_select_distinct(_t("SELECT DISTINCT a FROM t")) is False

def test_no_having():
    assert P.no_having(_t("SELECT a, COUNT(*) FROM t GROUP BY a")) is True
    assert P.no_having(_t("SELECT a, COUNT(*) FROM t GROUP BY a HAVING COUNT(*) > 1")) is False

def test_no_where():
    assert P.no_where(_t("SELECT a FROM t")) is True
    assert P.no_where(_t("SELECT a FROM t WHERE a > 1")) is False

def test_single_group_key():
    assert P.single_group_key(_t("SELECT a, COUNT(*) FROM t GROUP BY a")) is True
    assert P.single_group_key(_t("SELECT a, b FROM t GROUP BY a, b")) is False
    assert P.single_group_key(_t("SELECT a FROM t")) is False   # no GROUP BY at all


# --- segment / column guards ---

def test_columns_exist():
    seg = _FakeSeg({'a': {}, 'b': {}})
    assert P.columns_exist(seg, 'a') is True
    assert P.columns_exist(seg, 'a', 'b') is True
    assert P.columns_exist(seg, 'a', 'c') is False

def test_no_deleted_rows():
    assert P.no_deleted_rows(_FakeSeg({'a': {}}, presence=None)) is True
    assert P.no_deleted_rows(_FakeSeg({'a': {}}, presence=[True, False])) is False

def test_key_not_nullable():
    assert P.key_not_nullable(_FakeSeg({'a': {}}), 'a') is True
    assert P.key_not_nullable(_FakeSeg({'a': {'has_null': False}}), 'a') is True
    assert P.key_not_nullable(_FakeSeg({'a': {'has_null': True}}), 'a') is False
