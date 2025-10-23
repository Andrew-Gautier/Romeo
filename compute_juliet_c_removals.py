# Compute minimal removals for .90 and .95 similarity thresholds on juliet_c.db using similarity_hash.py
import os
import sqlite3
import pandas as pd
from similarity_hash import tokenize_code, simhash, build_pairs, greedy_vertex_cover

# Locate the juliet_c database
db_path = os.path.join('datasets', 'juliet_c.db')
assert os.path.exists(db_path), f"Database not found at {db_path}"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Find the first table that contains a 'code' column
code_table = None
code_col = 'code'
for (tbl_name,) in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
    cols = [row[1] for row in cursor.execute(f"PRAGMA table_info({tbl_name});").fetchall()]
    if code_col in cols:
        code_table = tbl_name
        break

if not code_table:
    conn.close()
    raise RuntimeError("Couldn't find a table with a 'code' column in juliet_c.db")

print(f"Using table: {code_table}, column: {code_col}")

# Fetch all code entries
query_all = f"SELECT rowid AS rid, {code_col} AS code FROM {code_table} WHERE {code_col} IS NOT NULL ORDER BY rowid"
codes_df = pd.read_sql_query(query_all, conn)
conn.close()

rids = codes_df['rid'].tolist()
codes = codes_df['code'].astype(str).tolist()
tokens_list = [tokenize_code(code) for code in codes]
hashes = [simhash(tokens) for tokens in tokens_list]


# Stream pairs to disk as CSV to avoid memory overload
import csv
pair_csv = 'pairs_output.csv'
with open(pair_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['rid_i','rid_j','hamming','similarity'])
    writer.writeheader()
    # Use bucketing to reduce comparisons
    num_buckets = 8
    threshold = 6
    bucket_shift = 64 - num_buckets
    from similarity_hash import hamming_distance
    buckets = {}
    for i, h in enumerate(hashes):
        bucket_key = h >> bucket_shift
        buckets.setdefault(bucket_key, []).append((rids[i], h))
    for bucket_hashes in buckets.values():
        for i in range(len(bucket_hashes)):
            rid_i, hash_i = bucket_hashes[i]
            for j in range(i + 1, len(bucket_hashes)):
                rid_j, hash_j = bucket_hashes[j]
                d = hamming_distance(hash_i, hash_j)
                if d <= threshold:
                    writer.writerow({
                        'rid_i': rid_i,
                        'rid_j': rid_j,
                        'hamming': d,
                        'similarity': 1 - d / 64.0,
                    })

# Now process the CSV in chunks for each threshold
for sim_threshold in [0.90, 0.95]:
    removed = set()
    edges_sim = {}
    adj = {}
    node_to_edges = {}
    # First pass: build edge and adjacency structures
    with open(pair_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = float(row['similarity'])
            if s > sim_threshold:
                u, v = int(row['rid_i']), int(row['rid_j'])
                e = (u, v) if u < v else (v, u)
                edges_sim[e] = s
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
                node_to_edges.setdefault(u, set()).add(e)
                node_to_edges.setdefault(v, set()).add(e)
    # Greedy vertex cover (degree)
    active_nodes = set(n for n, neigh in adj.items() if neigh)
    while edges_sim:
        best_node = max(active_nodes, key=lambda n: len(adj.get(n, ())))
        removed.add(best_node)
        for m in list(adj[best_node]):
            e = (best_node, m) if best_node < m else (m, best_node)
            edges_sim.pop(e, None)
            adj[m].discard(best_node)
            node_to_edges[m].discard(e)
        adj[best_node].clear()
        node_to_edges[best_node].clear()
        active_nodes = set(n for n, neigh in adj.items() if neigh)
    print(f"Similarity threshold {sim_threshold:.2f}: Records to remove = {len(removed)}")
    print(f"First 20 removed record IDs: {sorted(removed)[:20]}")
