import json
import os
from tqdm import tqdm


def load_id2meta(dataset):
    """Load ID to Metadata mapping"""
    path = f'./data/{dataset}_id2meta_text.json'
    print(f"Loading metadata from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_copurchase_neighbors(dataset):
    """Load co-purchase neighbors (numeric IDs)"""
    path = f'../data/{dataset}_collaborative_neighbors.json'
    print(f"Loading co-purchase neighbors from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_item(item_meta):
    """Format an item as Item text ID: [...] Title: ..."""
    summary_words = item_meta.get('summary_words', [])
    text = "Item keywords: [" + ", ".join(summary_words) + "]"
    if "title" in item_meta:
        text += f" Title: {item_meta['title']}.\n"
    else:
        text += " Title: None.\n"
    return text


NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def create_sft_sample(anchor_id, neighbor_ids, id2meta, top_k=7):
    """Create a single SFT training sample from co-purchase neighbors"""
    if anchor_id not in id2meta:
        return None

    anchor_meta = id2meta[anchor_id]

    # Filter valid neighbors and take top_k
    valid_neighbors = []
    for nid in neighbor_ids:
        nid_str = str(nid)
        if nid_str in id2meta:
            valid_neighbors.append(id2meta[nid_str])
        if len(valid_neighbors) >= top_k:
            break

    if not valid_neighbors:
        return None

    k_word = NUM_WORDS.get(top_k, str(top_k))
    instruction = (
        f"Given a target item in the format [keywords, title], recommend {k_word} items "
        "that are most likely to be co-purchased with it.\n"
        "Return the items sorted by likelihood, from most likely to least likely, "
        "and format each item as [keywords, title]."
    )

    input_text = format_item(anchor_meta)

    output_parts = []
    for i, meta in enumerate(valid_neighbors, 1):
        output_parts.append(format_item(meta))
    output_text = "".join(output_parts)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def main(dataset, top_k=7):
    id2meta = load_id2meta(dataset)
    neighbors = load_copurchase_neighbors(dataset)

    sft_data = []
    print(f"Generating co-purchase SFT data for {dataset} (top_k={top_k})...")

    for anchor_id, neighbor_ids in tqdm(neighbors.items()):
        sample = create_sft_sample(anchor_id, neighbor_ids, id2meta, top_k=top_k)
        if sample:
            sft_data.append(sample)

    output_dir = "./train_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset}_tid_collaborative_neighbors_sft.json"

    print(f"Saving {len(sft_data)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys", help="Dataset name (e.g., sports, toys, beauty, yelp)")
    parser.add_argument("--topk", type=int, default=5, help="Number of neighbors to recommend")
    args = parser.parse_args()
    main(args.dataset, top_k=args.topk)