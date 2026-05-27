import os
import json
import argparse
from tqdm import tqdm


def build_alignment_dataset(dataset):
    """Build alignment dataset: given item title, predict its codebook token triplet.
    
    Format:
        instruction: "Predict the item identifier for the following product title."
        input: "Air Jordan 1 Retro High OG"
        output: "<a_42><b_817><c_305>"
    """
    id2meta_file = f"./data/{dataset}_id2meta_codebook.json"
    output_file = f"./train_data/{dataset}_sid_alignment_sft.json"

    with open(id2meta_file, "r", encoding="utf-8") as f:
        id2meta = json.load(f)

    print(f"Loaded {len(id2meta)} items from {id2meta_file}")

    instruction = "Predict the item identifier for the following product title."

    samples = []
    for item_id, meta in tqdm(id2meta.items(), desc="Building alignment data"):
        title = meta.get("title", "").strip()
        if not title:
            continue

        summary_words = meta.get("summary_words", [])
        if len(summary_words) != 3:
            continue

        sid = "".join(f"<{w}>" for w in summary_words)

        samples.append({
            "instruction": instruction,
            "input": title,
            "output": sid,
        })

    # Save full dataset
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} alignment samples to {output_file}")

    # Show sample
    if samples:
        s = samples[0]
        print(f"\nSample:")
        print(f"  instruction: {s['instruction']}")
        print(f"  input: {s['input']}")
        print(f"  output: {s['output']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys")
    args = parser.parse_args()

    build_alignment_dataset(args.dataset)
