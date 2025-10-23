"""
similarity_hash.py

Core functions for SimHash-based code similarity analysis and greedy minimization of near-duplicate records.

Functions:
- tokenize_code: Tokenizes code into identifiers, numbers, and symbols.
- md5_64: 64-bit hash of a string using MD5.
- simhash: Computes SimHash fingerprint from tokens.
- hamming_distance: Bitwise Hamming distance between two integers.
- build_pairs: Generates all pairs with SimHash similarity above a threshold.
- greedy_vertex_cover: Finds a minimal set of records to remove so all pairs are below a similarity threshold.
"""
import re
import hashlib
from collections import Counter, defaultdict
import pandas as pd

def tokenize_code(code):
    """Tokenize code into identifiers, numbers, multi-char ops, and single symbols."""
    return re.findall(r"[A-Za-z_]\w+|\d+|==|!=|<=|>=|&&|\|\||[{}()\[\].,;:+\-*/%&|^!<>?=]", code)

def md5_64(text):
    """Return a 64-bit integer hash of text using MD5."""
    return int.from_bytes(hashlib.md5(text.encode('utf-8')).digest()[:8], 'big', signed=False)

def simhash(tokens, hashbits=64):
    """Compute SimHash fingerprint from a list of tokens."""
    v = [0] * hashbits
    for token, weight in Counter(tokens).items():
        h = md5_64(token)
        for i in range(hashbits):
            v[i] += weight if h & (1 << i) else -weight
    return sum((1 << i) if v[i] >= 0 else 0 for i in range(hashbits))

def hamming_distance(a, b):
    """Return the Hamming distance between two integers."""
    return (a ^ b).bit_count()

def build_pairs(rids, hashes, threshold, num_buckets=16):
    """Return DataFrame of all pairs with Hamming distance <= threshold."""
    if num_buckets > 16:
        num_buckets = 16  # Practical limit for 64-bit hashes
    
    bucket_shift = 64 - num_buckets
    buckets = defaultdict(list)
    
    # Group hashes by their first 'num_buckets' bits
    for i, h in enumerate(hashes):
        bucket_key = h >> bucket_shift
        buckets[bucket_key].append((rids[i], h))
    
    pairs = []
    # Only compare hashes within the same bucket
    for bucket_hashes in buckets.values():
        for i in range(len(bucket_hashes)):
            rid_i, hash_i = bucket_hashes[i]
            for j in range(i + 1, len(bucket_hashes)):
                rid_j, hash_j = bucket_hashes[j]
                d = hamming_distance(hash_i, hash_j)
                if d <= threshold:
                    pairs.append({
                        'rid_i': rid_i,
                        'rid_j': rid_j,
                        'hamming': d,
                        'similarity': 1 - d / 64.0,
                    })
    
    return pd.DataFrame(pairs)

def greedy_vertex_cover(pairs_df, sim_threshold, score_fn=None):
    """
    Greedily remove records so all pairs have similarity <= sim_threshold.
    score_fn: function(node, adj, edges_sim, node_to_edges) -> score (default: degree)
    Returns: set of removed record IDs
    """
    viol = pairs_df[pairs_df['similarity'] > sim_threshold]
    def edge_key(a, b): return (a, b) if a < b else (b, a)
    edges_sim = {}
    adj = defaultdict(set)
    for _, row in viol.iterrows():
        u, v, s = int(row['rid_i']), int(row['rid_j']), float(row['similarity'])
        e = edge_key(u, v)
        if e not in edges_sim or s > edges_sim[e]:
            edges_sim[e] = s
        adj[u].add(v)
        adj[v].add(u)
    node_to_edges = defaultdict(set)
    for (u, v) in edges_sim:
        node_to_edges[u].add((u, v))
        node_to_edges[v].add((u, v))
    if score_fn is None:
        score_fn = lambda n, adj, edges_sim, node_to_edges: len(adj.get(n, ()))
    removed = set()
    active_nodes = set(n for n, neigh in adj.items() if neigh)
    while edges_sim:
        best_node = max(active_nodes, key=lambda n: score_fn(n, adj, edges_sim, node_to_edges))
        removed.add(best_node)
        for m in list(adj[best_node]):
            e = edge_key(best_node, m)
            edges_sim.pop(e, None)
            adj[m].discard(best_node)
            node_to_edges[m].discard(edge_key(min(best_node, m), max(best_node, m)))
        adj[best_node].clear()
        node_to_edges[best_node].clear()
        active_nodes = set(n for n, neigh in adj.items() if neigh)
    return removed
