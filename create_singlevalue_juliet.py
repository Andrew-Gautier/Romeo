"""
create_singlevalue_juliet.py

For each Juliet database (C, Java, C#) in the datasets/ directory, create a
new database that contains only a single record per unique 'id' value.

Selection strategy: keep the record with the smallest internal rowid for each id.
Output databases are named with suffix '_id_singlevalue.db', e.g.:
    juliet_c_id_singlevalue.db
    juliet_java_id_singlevalue.db
    juliet_csharp_id_singlevalue.db

If a database does not contain a table with an 'id' column, it is skipped.
"""
import os
import shutil
import sqlite3
from datetime import datetime

DATASETS_DIR = 'datasets'
INPUT_DATABASES = [
    'juliet_c.db',
    'juliet_java.db',
    'juliet_csharp.db',
]

def find_id_tables(conn):
    """Return list of table names that have an 'id' column."""
    cursor = conn.cursor()
    tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    grp_tables = []
    for tbl in tables:
        cols = [c[1] for c in cursor.execute(f"PRAGMA table_info({tbl});").fetchall()]
        if 'id' in cols:
            grp_tables.append(tbl)
    return grp_tables

def reduce_table_to_single_id(conn, table):
    """Delete duplicate rows so only one (min rowid) per id remains."""
    cursor = conn.cursor()
    # Count before
    total = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    distinct = cursor.execute(f"SELECT COUNT(DISTINCT id) FROM {table}").fetchone()[0]
    print(f"  [INFO] Table '{table}': total rows={total}, distinct id={distinct}")

    # Perform deletion of all but min(rowid) per grp
    # Use a temporary table holding survivors to avoid correlated subquery performance issues on huge tables.
    print(f"  [DEBUG] Building survivor rowid list for table '{table}'...")
    cursor.execute("CREATE TEMP TABLE __survivors(rowid INTEGER PRIMARY KEY)")
    cursor.execute(f"INSERT INTO __survivors(rowid) SELECT MIN(rowid) FROM {table} GROUP BY id")
    survivors = cursor.execute("SELECT COUNT(*) FROM __survivors").fetchone()[0]
    print(f"  [DEBUG] Survivors computed: {survivors}")

    # Delete rows not in survivors
    print(f"  [DEBUG] Deleting duplicates for table '{table}'...")
    cursor.execute(f"DELETE FROM {table} WHERE rowid NOT IN (SELECT rowid FROM __survivors)")
    remaining = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  [INFO] Table '{table}' after reduction: rows={remaining}")

    # Drop temp table
    cursor.execute("DROP TABLE __survivors")

def process_database(input_path):
    base = os.path.basename(input_path)
    out_name = base.replace('.db', '_id_singlevalue.db')
    output_path = os.path.join(DATASETS_DIR, out_name)

    print(f"\n=== Processing database: {base} ===")
    if not os.path.exists(input_path):
        print(f"[WARN] Input database not found: {input_path}. Skipping.")
        return

    print(f"[DEBUG] Copying '{base}' to '{out_name}'...")
    shutil.copy(input_path, output_path)

    conn = sqlite3.connect(output_path)
    try:
        grp_tables = find_id_tables(conn)
        if not grp_tables:
            print(f"[WARN] No tables with 'id' column found in {base}. Nothing to do.")
            return
        print(f"[INFO] Tables with 'id': {grp_tables}")
        for tbl in grp_tables:
            reduce_table_to_single_id(conn, tbl)
        conn.commit()
        print(f"[SUCCESS] Wrote single-value database: {output_path}")
    finally:
        conn.close()

def main():
    print("=== JULIET SINGLE-VALUE REDUCTION ===")
    print(f"Start time: {datetime.now()}")
    for db in INPUT_DATABASES:
        process_database(os.path.join(DATASETS_DIR, db))
    print(f"End time: {datetime.now()}")

if __name__ == '__main__':
    main()
