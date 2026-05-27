"""
Extract aligned codebook token embeddings from a full alignment checkpoint.

Usage:
    python extract_aligned_embeddings.py --model_dir ./model_toys_codebook_align --num_new_tokens 3072 --output ./data/toys_codebook_aligned_embeddings.safetensors
    python extract_aligned_embeddings.py --model_dir ./model_sports_codebook_align --num_new_tokens 3072 --output ./data/sports_codebook_aligned_embeddings.safetensors
    python extract_aligned_embeddings.py --model_dir ./model_beauty_codebook_align --num_new_tokens 3072 --output ./data/beauty_codebook_aligned_embeddings.safetensors
"""
import argparse
import glob
import json
import os

import torch
from safetensors.torch import load_file, save_file


def detect_num_new_tokens(dataset):
    """Auto-detect number of new tokens from the special_tokens.txt file."""
    token_file = f"./data/{dataset}_special_tokens.txt"
    if os.path.exists(token_file):
        with open(token_file) as f:
            tokens = f.read().strip()
        if tokens:
            return len(tokens.split(","))
    return None


def extract_embeddings(model_dir, num_new_tokens, output_path, dataset="toys"):
    # Find the shard containing embed_tokens
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        # Find which shard has embed_tokens
        embed_shard = None
        for key, shard in index["weight_map"].items():
            if "embed_tokens" in key:
                embed_shard = os.path.join(model_dir, shard)
                embed_key = key
                break
    else:
        # Single shard
        embed_shard = os.path.join(model_dir, "model.safetensors")
        embed_key = None

    # Load the shard
    state = load_file(embed_shard, device="cpu")

    # Find the embed_tokens key
    if embed_key is None:
        for key in state.keys():
            if "embed_tokens" in key:
                embed_key = key
                break

    if embed_key is None:
        raise RuntimeError(f"Could not find embed_tokens weight in {embed_shard}")

    embed = state[embed_key]
    print(f"Full embedding shape: {embed.shape}")

    # Auto-detect num_new_tokens if not specified
    if num_new_tokens is None:
        num_new_tokens = detect_num_new_tokens(dataset)
        if num_new_tokens is None:
            raise RuntimeError("Could not auto-detect num_new_tokens. Specify --num_new_tokens manually.")
        print(f"Auto-detected num_new_tokens: {num_new_tokens} (from data/{dataset}_special_tokens.txt)")
    
    print(f"Extracting last {num_new_tokens} rows (new codebook tokens)")

    new_embeds = embed[-num_new_tokens:]
    print(f"Extracted shape: {new_embeds.shape}, dtype: {new_embeds.dtype}")

    save_file({"codebook_embeddings": new_embeds}, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved to: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys", help="Dataset name (toys, sports, beauty)")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to alignment checkpoint (default: ./model_{dataset}_codebook_align)")
    parser.add_argument("--num_new_tokens", type=int, default=None, help="Number of new special tokens (auto-detected if not specified)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: ./data/{dataset}_codebook_aligned_embeddings.safetensors)")
    args = parser.parse_args()

    model_dir = args.model_dir or f"./model_{args.dataset}_codebook_align"
    output_path = args.output or f"./data/{args.dataset}_codebook_aligned_embeddings.safetensors"

    extract_embeddings(model_dir, args.num_new_tokens, output_path, dataset=args.dataset)
