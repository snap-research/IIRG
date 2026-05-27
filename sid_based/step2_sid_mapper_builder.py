import json
import pickle
import argparse
from tqdm import tqdm


PREFIXES = ["a", "b", "c"]
CODEBOOK_SIZES = [1024, 512, 256]  # Must match codebook_generation_v2.py --codebook_sizes


def build_codebook_id2meta(dataset):
    """Build id2meta with codebook-based special tokens instead of language keywords.
    
    Each item is represented as <a_X><b_Y><c_Z> where X, Y, Z are codebook indices.
    """
    codebook_file = f"./data/{dataset}_codebook.pickle"
    item_file = f"../data/{dataset}.item.json"
    output_file = f"./data/{dataset}_id2meta_codebook.json"
    token_file = f"./data/{dataset}_special_tokens.json"

    # Load codebook codes
    with open(codebook_file, "rb") as f:
        codes = pickle.load(f)
    print(f"Loaded codebook codes: {codes.shape}")  # [N+1, 3], row 0 is padding

    n_items = codes.shape[0] - 1  # exclude padding row
    n_layers = codes.shape[1]
    assert n_layers == len(PREFIXES), f"Expected {len(PREFIXES)} layers, got {n_layers}"

    # Load item metadata directly (no dependency on TID pipeline)
    with open(item_file, "r", encoding="utf-8") as f:
        id2meta = json.load(f)
    print(f"Loaded {len(id2meta)} items from {item_file}")

    # Generate ALL possible special tokens (full codebook range, not just used ones)
    all_special_tokens = set()
    for l, K in enumerate(CODEBOOK_SIZES):
        for idx in range(K):
            all_special_tokens.add(f"<{PREFIXES[l]}_{idx}>")

    # Build new id2meta with codebook tokens
    new_id2meta = {}
    for item_id, meta in tqdm(id2meta.items(), desc="Building codebook id2meta"):
        idx = int(item_id)  # id2meta is 1-indexed, codes row 0 is padding
        if idx < 1 or idx > n_items:
            print(f"Warning: item_id={item_id} out of range, skipping")
            continue

        item_codes = codes[idx].tolist()
        token_list = [f"{PREFIXES[l]}_{item_codes[l]}" for l in range(n_layers)]
        sid = "".join(f"<{t}>" for t in token_list)

        new_meta = dict(meta)
        new_meta["summary_words"] = token_list
        new_meta["sid"] = sid
        new_id2meta[item_id] = new_meta

    # Save new id2meta
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_id2meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(new_id2meta)} items to {output_file}")

    # Save sorted special tokens list
    sorted_tokens = sorted(all_special_tokens, key=lambda t: (t.split("_")[0], int(t.split("_")[1].rstrip(">"))))
    with open(token_file, "w", encoding="utf-8") as f:
        json.dump(sorted_tokens, f, indent=2)
    print(f"Saved {len(sorted_tokens)} special tokens to {token_file}")

    # Save comma-separated tokens for CLI usage (--add_special_tokens)
    token_txt_file = f"./data/{dataset}_special_tokens.txt"
    with open(token_txt_file, "w") as f:
        f.write(",".join(sorted_tokens))
    print(f"Saved comma-separated tokens to {token_txt_file}")

    # Print summary
    for l in range(n_layers):
        unique_codes = len(set(codes[:, l].tolist()))
        print(f"  Layer {PREFIXES[l]}: {unique_codes} unique tokens")

    # Show a sample
    sample_id = list(new_id2meta.keys())[0]
    sample = new_id2meta[sample_id]
    print(f"\nSample (id={sample_id}):")
    print(f"  title: {sample.get('title', 'N/A')[:80]}")
    print(f"  sid: {sample['sid']}")
    print(f"  summary_words: {sample['summary_words']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys")
    args = parser.parse_args()

    build_codebook_id2meta(args.dataset)