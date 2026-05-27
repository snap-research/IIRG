import argparse
import json
import pickle
import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_generation_outputs(gen_file):
    """Load pre-generated outputs from JSON file"""
    with open(gen_file, 'r', encoding='utf-8') as f:
        outputs = json.load(f)
    return outputs


def load_item_frequency(freq_file):
    """Load item frequency from pickle file"""
    with open(freq_file, 'rb') as f:
        item_freq = pickle.load(f)
    return item_freq


def _extract_seen_iids(sample):
    """Extract observed item IDs from a sample's item_id_list (excluding test GT).
    At test time, the user has seen: train items + valid GT. Only test GT is unseen."""
    item_id_list = sample.get("item_id_list", [])
    # Exclude last item only (test_ground_truth) — valid GT is part of input
    return set(item_id_list[:-1])


def load_tid_mappings(dataset_name):
    """Load codebook tid to item id mapping"""
    tid2item_id_path = f'./train_data/item_id2tid/{dataset_name}_tid2item_id.json'
    with open(tid2item_id_path, 'r', encoding='utf-8') as f:
        tid2item_id = json.load(f)
    return tid2item_id


def parse_codebook_tokens(content):
    """Parse codebook tokens from generated content.
    
    Extracts sequences like <a_X><b_Y><c_Z> from raw generated text.
    Returns the tid key (comma-separated) for lookup, e.g. "a_42,b_817,c_305"
    """
    # Match patterns like <a_123><b_456><c_789>
    tokens = re.findall(r'<([abc]_\d+)>', content)
    if len(tokens) >= 3:
        # Group into triplets
        triplets = []
        for i in range(0, len(tokens) - 2, 3):
            triplet = tokens[i:i+3]
            # Validate order: a, b, c
            if triplet[0].startswith('a_') and triplet[1].startswith('b_') and triplet[2].startswith('c_'):
                triplets.append(triplet)
        return triplets
    return []


def get_iid_by_codebook_tid(triplet, tid2item_id):
    """Get item IDs by codebook TID triplet.
    
    Args:
        triplet: list of 3 tokens, e.g. ["a_42", "b_817", "c_305"]
        tid2item_id: mapping from comma-separated tid key to item IDs
    
    Returns:
        list of item IDs
    """
    tid_key = ",".join(triplet)
    if tid_key in tid2item_id:
        return tid2item_id[tid_key]
    return []


def get_top_k_items_by_frequency(iid_list, item_freq, k=10):
    """Get top-k items by purchase frequency from a list of item IDs"""
    iid_freq_list = [(iid, item_freq.get(iid, 0)) for iid in iid_list]
    iid_freq_list.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in iid_freq_list[:k]]


def process_generation_outputs(gen_outputs, tid2item_id, item_freq, rerank_by_popularity=False):
    """Process generation outputs for codebook-based approach"""
    results = []

    for sample in tqdm(gen_outputs, desc="Processing generation outputs"):
        dic = sample.copy()
        iid_gt = sample.get("test_ground_truth_id")
        seen_iids = _extract_seen_iids(sample)

        # Extract codebook triplets from raw_contents
        all_triplets = []
        if "raw_contents" in sample:
            for raw_content in sample["raw_contents"]:
                triplets = parse_codebook_tokens(raw_content)
                for triplet in triplets:
                    if triplet not in all_triplets:
                        all_triplets.append(triplet)

        dic["contents_len"] = len(all_triplets)

        # For each unique triplet, get candidate items
        iids = []
        all_results = []

        for i, triplet in enumerate(all_triplets):
            candidate_iids = get_iid_by_codebook_tid(triplet, tid2item_id)

            # Remove items observed in the input sequence
            if seen_iids:
                candidate_iids = [iid for iid in candidate_iids if iid not in seen_iids]

            # Get top-1 by frequency
            top_iids = get_top_k_items_by_frequency(candidate_iids, item_freq, k=1)

            all_results.append({
                'sequence_id': i,
                'triplet': triplet,
                'tid_key': ",".join(triplet),
                'iid': top_iids
            })

            iids.extend(top_iids)

        # Remove duplicates while preserving order
        ids = []
        for i in iids:
            if i not in ids:
                ids.append(i)
        iids = ids[:20]

        # Pad if fewer than 20 items
        if len(iids) < 20:
            for triplet in all_triplets:
                candidate_iids = get_iid_by_codebook_tid(triplet, tid2item_id)
                if seen_iids:
                    candidate_iids = [iid for iid in candidate_iids if iid not in seen_iids]
                for iid in candidate_iids:
                    if iid not in iids:
                        iids.append(iid)
                    if len(iids) >= 20:
                        break
                if len(iids) >= 20:
                    break

        iids = iids[:20]

        # Optionally re-rank top-5 by popularity
        if rerank_by_popularity:
            top5 = sorted(iids[:5], key=lambda x: item_freq.get(x, 0), reverse=True)
            iids = top5 + iids[5:]

        dic["all_results"] = all_results
        dic["iids"] = iids
        dic["iids_len"] = len(iids)
        dic["iid_gt"] = iid_gt

        results.append(dic)

    return results


def calculate_recall(results):
    """Calculate recall and NDCG metrics from results"""
    scores = [0] * 20
    dcg = [0.0, 0.0, 0.0, 0.0]  # DCG for @1, @5, @10, @20

    for result in results:
        iid_gt = result.get("iid_gt")
        iids = result.get("iids", [])

        for i, iid in enumerate(iids):
            if i < len(scores) and iid_gt == iid:
                scores[i] += 1
                dcg_value = 1.0 / np.log2(i + 2)
                if i < 1:
                    dcg[0] += dcg_value
                if i < 5:
                    dcg[1] += dcg_value
                if i < 10:
                    dcg[2] += dcg_value
                if i < 20:
                    dcg[3] += dcg_value
                break

    total_samples = len(results)
    recall_metrics = {}
    recall_metrics["recall@1"] = sum(scores[:1]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@5"] = sum(scores[:5]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@10"] = sum(scores[:10]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@20"] = sum(scores[:20]) / total_samples if total_samples > 0 else 0

    if total_samples > 0:
        recall_metrics["ndcg@1"] = dcg[0] / total_samples
        recall_metrics["ndcg@5"] = dcg[1] / total_samples
        recall_metrics["ndcg@10"] = dcg[2] / total_samples
        recall_metrics["ndcg@20"] = dcg[3] / total_samples
    else:
        recall_metrics["ndcg@1"] = 0.0
        recall_metrics["ndcg@5"] = 0.0
        recall_metrics["ndcg@10"] = 0.0
        recall_metrics["ndcg@20"] = 0.0

    return recall_metrics, scores, dcg


def main(gen_output_file, item_freq_file, dataset_name, rerank_by_popularity=False):
    """Main evaluation function"""

    # Load tid mappings
    print("Loading codebook TID mappings...")
    tid2item_id = load_tid_mappings(dataset_name)
    print(f"Loaded {len(tid2item_id)} unique TID keys")

    # Load pre-generated outputs
    print(f"Loading generation outputs from {gen_output_file}...")
    gen_outputs = load_generation_outputs(gen_output_file)
    # gen_outputs = [s for s in gen_outputs if 1 <= int(s.get("metadata", {}).get("user_id", -1)) <= 3000]
    print(f"Loaded {len(gen_outputs)} generation outputs")

    # Load item frequency
    print(f"Loading item frequency from {item_freq_file}...")
    item_freq = load_item_frequency(item_freq_file)
    print(f"Loaded frequency for {len(item_freq)} items")

    # Process generation outputs
    print("Processing outputs...")
    results = process_generation_outputs(
        gen_outputs, tid2item_id, item_freq,
        rerank_by_popularity=rerank_by_popularity
    )
    print(f"Processed {len(results)} results")

    # Calculate metrics
    print("Calculating metrics...")
    recall_metrics, scores, dcg = calculate_recall(results)

    # Output results
    print("\n" + "=" * 50)
    print("Evaluation Results (Codebook)")
    print("=" * 50)
    for metric, value in recall_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save detailed results
    output_file = gen_output_file.replace('.json', '_eval.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': recall_metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_file}")

    return recall_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sports")
    parser.add_argument("--rerank_by_popularity", action="store_true",
                        help="Re-rank the top-5 candidates by item popularity")
    args = parser.parse_args()

    dataset_name = args.dataset
    gen_output_file = f"./results/{dataset_name}_generation_outputs_sid.json"
    item_freq_file = f"../data/{dataset_name}_item_freq.pkl"

    main(gen_output_file, item_freq_file, dataset_name, rerank_by_popularity=args.rerank_by_popularity)
