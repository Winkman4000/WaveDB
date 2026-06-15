"""
commands.py -- THE MUTATION/INIT ROUTER.

The write side of the spine: DDL and DML statements (CREATE / ALTER / INSERT / DROP /
DELETE / UPDATE) -- how the database gets initialized and mutated, as opposed to the
query layer (controller.py + read_methods.py) which only reads.

route(db, sql, tree) sends a mutation to its handler and returns the result, or the
_NOT_A_COMMAND sentinel if `tree` is a query (SELECT) so the dispatcher proceeds to
the read path. Like controller, this holds no logic of its own -- it routes to the
handlers in wdb_ddl / wdb_dml.
"""
import wdb_ddl
import wdb_dml
import wdb_sql

E = wdb_sql.E

_NOT_A_COMMAND = object()   # sentinel: this statement is not a mutation -> caller reads


def route(db, sql, tree):
    """Route a DDL/DML statement to its handler. Returns the handler's result (which may
    legitimately be None, e.g. DROP), or _NOT_A_COMMAND when `tree` is not a mutation."""
    if isinstance(tree, E.Create):
        return wdb_ddl.create_table(db.cat, sql)
    if isinstance(tree, E.Alter):
        return wdb_ddl.alter_table(db.cat, sql, db)
    if isinstance(tree, E.Insert):
        return wdb_dml.insert(db.cat, sql)
    if isinstance(tree, E.Drop):
        db.cat.drop_table(tree.this.this.name)
        return None
    if isinstance(tree, E.Delete):
        return wdb_dml.delete(db.cat, sql)
    if isinstance(tree, E.Update):
        return wdb_dml.update(db.cat, sql)
    return _NOT_A_COMMAND
