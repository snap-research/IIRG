"""
Build co-purchase neighbors and semantic neighbors for the Snap dataset.

1. Co-purchase neighbors (own): sliding-window co-occurrence from user sequences.
2. Semantic neighbors: batched cosine similarity (memory-efficient for million-scale items).

For testing, embeddings are replaced with random tensors.
"""

import os
import json
import torch
import pickle
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="beauty", help="Dataset name (e.g., sports, toys, beauty, yelp)")
args = parser.parse_args()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET = args.dataset

# ── Hyperparameters ──
WINDOW_SIZE = 5
TOPK = 10 # Co-purchase neighbors to keep
SEM_K = 500 # Semantic neighbors to keep / This should be large for TID generation
SEM_BATCH_SIZE = 1024  # rows processed at a time for similarity


def load_sequential_data(path):
    seq_data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            user_idx = int(parts[0])
            item_indices = [int(x) for x in parts[1:]]
            seq_data[user_idx] = item_indices
    return seq_data


# ────────────────────────────────────────────────
# 1. Co-purchase neighbors (own)
# ────────────────────────────────────────────────
def build_copurchase_neighbors(seq_data, num_items):
    """Sliding-window co-purchase neighbors.
    For each item at position i, neighbors are items within WINDOW_SIZE
    positions in both directions."""
    print("=== Building co-purchase neighbors ===")

    neighbor_counts = defaultdict(lambda: defaultdict(int))

    for user_id, items in tqdm(seq_data.items(), desc="Building windowed co-purchase"):
        trimmed = items[:-1]  # exclude last item (test data leakage prevention)  <- very important
        n = len(trimmed)
        for i in range(n):
            lower = max(0, i - WINDOW_SIZE)
            upper = min(n, i + WINDOW_SIZE + 1)
            for j in range(lower, upper):
                if j == i:  # skip self
                    continue
                if trimmed[i] != trimmed[j]:
                    neighbor_counts[trimmed[i]][trimmed[j]] += 1

    # Purchase frequency for tie-breaking
    purchase_freq = Counter()
    for items in seq_data.values():
        purchase_freq.update(items[:-1]) ## Exclude last item (test data leakage prevention)

    # Rank and select TOPK
    topk_neighbors = {}
    for item_id in range(1, num_items + 1):
        neigh = neighbor_counts.get(item_id, {})
        ranked = sorted(
            neigh.items(),
            key=lambda x: (-x[1], -purchase_freq.get(x[0], 0), x[0]),
        )[:TOPK]
        topk_neighbors[item_id] = [nid for nid, _ in ranked]

    n_with = sum(1 for v in topk_neighbors.values() if len(v) > 0)
    n_full = sum(1 for v in topk_neighbors.values() if len(v) >= TOPK)
    print(f"Items with any co-purchase neighbors: {n_with}/{num_items}")
    print(f"Items with >= {TOPK} neighbors: {n_full}/{num_items}")

    return topk_neighbors


# ────────────────────────────────────────────────
# 2. Semantic neighbors (batched)
# ────────────────────────────────────────────────
def build_semantic_neighbors(embeddings, batch_size=SEM_BATCH_SIZE):
    """
    Compute top-K semantic neighbors using batched cosine similarity.
    Processes `batch_size` query rows at a time to avoid OOM on million-scale items.
    Embeddings should already be L2-normalized.
    """
    print(f"\n=== Building semantic neighbors (batch_size={batch_size}) ===")
    n = embeddings.shape[0]
    num_batches = (n + batch_size - 1) // batch_size
    print(f"Total items: {n}, Batches: {num_batches}")
    all_topk_indices = []

    for start in tqdm(range(0, n, batch_size), desc="Semantic neighbor batches"):
        end = min(start + batch_size, n)
        # (batch, dim) @ (dim, n) -> (batch, n)
        sim = embeddings[start:end] @ embeddings.T
        # Zero out self-similarity and padding row
        sim[:, 0] = -1.0
        for i in range(end - start):
            sim[i, start + i] = -1.0
        topk = torch.topk(sim, k=SEM_K + 1, dim=1).indices  # +1 to account for potential self in batch
        all_topk_indices.append(topk)
        batch_num = start // batch_size + 1
        print(f"  Batch {batch_num}/{num_batches} | rows [{start}:{end}] | "
              f"sim matrix: ({end - start}, {n}) | "
              f"max sim: {sim.max().item():.4f} | min sim: {sim.min().item():.4f}")

    max_indices = torch.cat(all_topk_indices, dim=0)  # (n, SEM_K)

    # Convert to dict (skip index 0 which is a placeholder)
    topk_neighbors_semantic = {}
    for item_id in range(1, n):
        nbs = max_indices[item_id].tolist()
        # Filter out index 0 (placeholder) and self
        nbs = [nb for nb in nbs if nb != 0 and nb != item_id]
        topk_neighbors_semantic[item_id] = nbs[:SEM_K]

    print(f"Semantic neighbors built for {len(topk_neighbors_semantic)} items")
    return topk_neighbors_semantic


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
print(f"Loading sequential data for {DATASET}...")
seq_path = os.path.join(DATA_DIR, f"{DATASET}_sequential_data.txt")
seq_data = load_sequential_data(seq_path)
print(f"Users: {len(seq_data)}")

# Determine number of items
all_items = set()
for items in seq_data.values():
    all_items.update(items)
num_items = max(all_items)
print(f"Items: {num_items}")

# 1. Co-purchase neighbors

copurchase_neighbors = build_copurchase_neighbors(seq_data, num_items)

# 2. Semantic neighbors
# TODO: replace random tensor with actual embeddings once ready

emb_path = os.path.join(DATA_DIR, f"{DATASET}_embeddings.pt")
if os.path.exists(emb_path):
    print(f"\nLoading embeddings from {emb_path}...")
    X = torch.load(emb_path)
else:
    print(f"\nEmbeddings not found. Using random tensors for testing (n={num_items + 1}, dim=4096)...")
    raise NotImplementedError("Embeddings not found. Please run step1_embedding_builder.py first.")

# L2 normalize
X = torch.nn.functional.normalize(X, p=2, dim=1)
semantic_neighbors = build_semantic_neighbors(X)

# Backfill co-purchase neighbors: for items with < TOPK, use co-purchase neighbors
# of their semantically most similar items
for item_id in range(1, num_items + 1):
    cop = copurchase_neighbors.get(item_id, [])
    if len(cop) >= TOPK:
        continue
    sem = semantic_neighbors.get(item_id, [])
    existing = set(cop) | {item_id}
    for sem_nb in sem:
        if len(cop) >= TOPK:
            break
        # Borrow co-purchase neighbors from this semantic neighbor
        sem_nb_cop = copurchase_neighbors.get(sem_nb, [])
        for nb in sem_nb_cop:
            if nb not in existing:
                cop.append(nb)
                existing.add(nb)
            if len(cop) >= TOPK:
                break
    copurchase_neighbors[item_id] = cop

n_backfilled = sum(1 for item_id in range(1, num_items + 1)
                   if len(copurchase_neighbors.get(item_id, [])) > 0)
print(f"\nAfter backfill: {n_backfilled}/{num_items} items have neighbors")

# Save
cop_path = os.path.join(DATA_DIR, f"{DATASET}_collaborative_neighbors.json")
sem_path = os.path.join(DATA_DIR, f"{DATASET}_semantic_neighbors.json")

with open(cop_path, "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in copurchase_neighbors.items()}, f)
print(f"\nSaved collaborative neighbors to {cop_path}")

with open(sem_path, "w", encoding="utf-8") as f:
    json.dump({str(k): v for k, v in semantic_neighbors.items()}, f)
print(f"Saved semantic neighbors to {sem_path}")

# 3. Item frequency
purchase_freq = Counter()
for items in seq_data.values():
    purchase_freq.update(items[:-1]) ## Prevent test leakage by excluding last item

freq_path = os.path.join(DATA_DIR, f"{DATASET}_item_freq.pkl")
with open(freq_path, "wb") as f:
    pickle.dump(dict(purchase_freq), f)
print(f"Saved item frequencies ({len(purchase_freq)} items) to {freq_path}")

print("\nDone!")
