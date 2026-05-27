import argparse
import json
from pathlib import Path


def load_json(path: Path):
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    yield obj
            else:
                raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge TID SFT dataset files with per-source loss weights."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. sports")
    parser.add_argument("--input_dir", type=str, default="./train_data", help="Directory containing input files")
    parser.add_argument("--output_dir", type=str, default="./train_data", help="Directory to write merged output file")
    parser.add_argument("--copurchase_weight", type=float, default=1.0, help="Loss weight for copurchase samples")
    parser.add_argument("--semantic_weight", type=float, default=1.0, help="Loss weight for semantic samples")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="Loss weight for next-item prediction samples")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = [
        (input_dir / f"{args.dataset}_tid_collaborative_neighbors_sft.json", args.copurchase_weight),
        (input_dir / f"{args.dataset}_tid_semantic_neighbors_sft.json", args.semantic_weight),
        (input_dir / f"{args.dataset}_tid_next_item_prediction_sft_simplified.json", args.rec_weight),
    ]

    out_path = output_dir / f"{args.dataset}_tid_merged_sft.jsonl"

    count = 0
    weight_counts = {}
    with out_path.open("w", encoding="utf-8") as fout:
        for path, weight in input_files:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            n = 0
            for obj in load_json(path):
                obj["weight"] = weight
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1

            weight_counts[path.name] = (n, weight)
            count += n

    print(f"Merged {count} examples into {out_path}")
    for fname, (n, w) in weight_counts.items():
        print(f"  {fname}: {n} samples, weight={w}")


if __name__ == "__main__":
    main()
