import json
import os
from tqdm import tqdm


def format_sid(summary_words):
    """Format summary_words into [<a_X><b_Y><c_Z>]."""
    return "[" + "".join(f"<{w}>" for w in summary_words) + "]"


def load_id2meta(dataset):
    """Load ID to Metadata mapping (codebook version)"""
    path = f'./data/{dataset}_id2meta_codebook.json'
    print(f"Loading metadata from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_semantic_neighbors(dataset):
    """Load semantic neighbors (numeric IDs)"""
    path = f'../data/{dataset}_semantic_neighbors.json'
    print(f"Loading semantic neighbors from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_item(item_meta):
    """Format an item as Item keywords: [<a_X><b_Y><c_Z>] Title: ..."""
    summary_words = item_meta.get('summary_words', [])
    text = f"Item keywords: {format_sid(summary_words)}"
    if "title" in item_meta:
        text += f" Title: {item_meta['title']}.\n"
    else:
        text += " Title: None.\n"
    return text


def create_sft_sample(anchor_id, neighbor_ids, id2meta, top_k=5):
    """Create a single SFT training sample from semantic neighbors"""
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

    instruction = (
        "Given a target item in the format [keywords, title], list five items "
        "that are most semantically similar to it.\n"
        "Return the items sorted by similarity, from most similar to least similar, "
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


def main(dataset):
    id2meta = load_id2meta(dataset)
    neighbors = load_semantic_neighbors(dataset)

    sft_data = []
    print(f"Generating semantic SFT data for {dataset}...")

    for anchor_id, neighbor_ids in tqdm(neighbors.items()):
        sample = create_sft_sample(anchor_id, neighbor_ids, id2meta)
        if sample:
            sft_data.append(sample)

    output_dir = "./train_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset}_sid_semantic_neighbors_sft.json"

    print(f"Saving {len(sft_data)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys", help="Dataset name (e.g., sports, toys, beauty, yelp)")
    args = parser.parse_args()
    main(args.dataset)
