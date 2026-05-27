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
    """Extract observed item IDs from a sample (best-effort)."""
    candidate_keys = [
        "input_sequence",
        "user_sequence",
        "sequence",
        "history",
        "history_items",
        "seq_items",
        "items",
    ]
    for key in candidate_keys:
        if key not in sample:
            continue
        value = sample.get(key)
        if isinstance(value, list):
            seen = []
            for v in value:
                try:
                    seen.append(int(v))
                except Exception:
                    continue
            return set(seen)
        if isinstance(value, str):
            nums = re.findall(r"\d+", value)
            return set(int(v) for v in nums)

    # Fallback: try to parse prompt for an ID sequence if present
    prompt = sample.get("prompt")
    if isinstance(prompt, str):
        nums = re.findall(r"\d+", prompt)
        if nums:
            return set(int(v) for v in nums)
    return set()

def create_reverse_mapping(original_dict):
    """Create reverse mapping, split key into word list"""
    reverse_mapping = {}
    word_to_keys = defaultdict(list)
    
    for key_str, ids in original_dict.items():
        # Clean and split keywords
        words = [word.strip().lower() for word in key_str.split(',')]
        reverse_mapping[key_str] = {
            'words': words,
            'ids': ids
        }
        
        # Build index for each word
        for word in words:
            word_to_keys[word].append(key_str)
    
    return reverse_mapping, word_to_keys

def get_iid_by_tid(content, tid2item_id, reverse_mapping, word_to_keys):
    """Get item ID by TID, supports fuzzy matching"""
    threshold = 0
    iids = []
    tids = content.replace("[","").replace("]","").split(", ")
    tid_key = ",".join(tids)
    if tid_key in tid2item_id:
        iids.extend(tid2item_id[tid_key])
    else:
        # Fuzzy matching
        candidate_scores = defaultdict(float)
        query_words = tids
        for i, query_word in enumerate(query_words):
            # Position weight: words at front are more important
            position_weight = 1.0 / (i + 1)
            
            # Find candidates containing current query word
            for candidate_word, candidate_keys in word_to_keys.items():
                # Calculate similarity
                similarity = 0.0
                if query_word == candidate_word:
                    similarity = 1.0  # Exact match
                elif query_word in candidate_word or candidate_word in query_word:
                    similarity = 0.8  # Partial match
                # If similarity exceeds threshold, add score to all related candidates
                if similarity > 0:
                    for candidate_key in candidate_keys:
                        candidate_scores[candidate_key] += similarity * position_weight
        
        # Sort by score and filter
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Apply threshold
        for candidate_key, score in sorted_candidates:
            if score >= threshold:
                iids.extend(reverse_mapping[candidate_key]['ids'])
        
        iids = iids[:1]
    return iids

def get_top_k_items_by_frequency(iid_list, item_freq, k=1):
    """Get top-k items by purchase frequency from a list of item IDs"""
    # Create list of (iid, frequency) tuples
    iid_freq_list = []
    for iid in iid_list:
        freq = item_freq.get(iid, 0)
        iid_freq_list.append((iid, freq))
    
    # Sort by frequency (descending)
    iid_freq_list.sort(key=lambda x: x[1], reverse=True)
    
    # Return top-k item IDs
    return [iid for iid, _ in iid_freq_list[:k]]

def process_generation_outputs(gen_outputs, tid2item_id, reverse_mapping, word_to_keys, item_freq, rerank_by_popularity=False):
    """Process generation outputs and apply top-2 frequency sampling"""
    results = []
    
    for sample in tqdm(gen_outputs, desc="Processing generation outputs"):
        dic = sample.copy()
        iid_gt = sample.get("test_ground_truth_id")
        seen_iids = _extract_seen_iids(sample)
        
        # Extract contents (TIDs) from raw_contents
        contents = []
        if "raw_contents" in sample:
            for raw_content in sample["raw_contents"]:
                pattern = r'\[(.*?)\]'
                cons = re.findall(pattern, raw_content)
                for c in cons:
                    content_str = "[" + c + "]"
                    if content_str not in contents:
                        contents.append(content_str)
        
        dic["contents_len"] = len(contents)
        
        # For each unique TID (content), get top-2 items by frequency
        iids = []
        all_results = []
        
        for i, content in enumerate(contents):
            # Get all items with this TID
            candidate_iids = get_iid_by_tid(content, tid2item_id, reverse_mapping, word_to_keys)

            # Remove items observed in the input sequence
            if seen_iids:
                candidate_iids = [iid for iid in candidate_iids if iid not in seen_iids]
            
            # Sample top-2 by frequency
            top_iids = get_top_k_items_by_frequency(candidate_iids, item_freq, k=1)
            
            all_results.append({
                'sequence_id': i,
                'content': content,
                'iid': top_iids
            })
            
            # Extend main iids list
            iids.extend(top_iids)
        
        # Remove duplicates while preserving order
        ids = []
        for i in iids:
            if i not in ids:
                ids.append(i)
        
        # Take top-20
        iids = ids[:20]
        
        # If we have fewer than 20 items, pad with more candidates
        if len(iids) < 20:
            for i, content in enumerate(contents):
                candidate_iids = get_iid_by_tid(content, tid2item_id, reverse_mapping, word_to_keys)
                if seen_iids:
                    candidate_iids = [iid for iid in candidate_iids if iid not in seen_iids]
                # Get all items (not just top-2) for padding
                for iid in candidate_iids:
                    if iid not in iids:
                        iids.append(iid)
                    if len(iids) >= 20:
                        break
                if len(iids) >= 20:
                    break
        
        iids = iids[:20]
        
        # Optionally re-rank the top-5 candidates by popularity (descending frequency)
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
        
        # Check if ground truth is in predictions
        for i, iid in enumerate(iids):
            if i < len(scores) and iid_gt == iid:
                scores[i] += 1
                # Compute DCG: 1 / log2(i+2) because position is 0-indexed
                dcg_value = 1.0 / np.log2(i + 2)
                # Add to appropriate cutoff levels
                if i < 1:
                    dcg[0] += dcg_value
                if i < 5:
                    dcg[1] += dcg_value
                if i < 10:
                    dcg[2] += dcg_value
                if i < 20:
                    dcg[3] += dcg_value
                break
    
    # Calculate metrics
    total_samples = len(results)
    recall_metrics = {}
    recall_metrics["recall@1"] = sum(scores[:1]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@5"] = sum(scores[:5]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@10"] = sum(scores[:10]) / total_samples if total_samples > 0 else 0
    recall_metrics["recall@20"] = sum(scores[:20]) / total_samples if total_samples > 0 else 0
    
    # NDCG: For a single relevant item per query, IDCG = 1.0 (when relevant item is at position 0)
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

def load_tid_mappings(dataset_name):
    """Load tid to item id and reverse mappings for a dataset"""
    tid2item_id_path = f'./train_data/item_id2tid/{dataset_name}_tid2item_id.json'
    with open(tid2item_id_path, 'r', encoding='utf-8') as f:
        tid2item_id = json.load(f)
    
    reverse_mapping, word_to_keys = create_reverse_mapping(tid2item_id)
    num_keywords = len([w.strip() for w in next(iter(tid2item_id)).split(',')])
    
    return tid2item_id, reverse_mapping, word_to_keys, num_keywords

def main(gen_output_file, item_freq_file, dataset_name, rerank_by_popularity=False):
    """Main evaluation function"""
    
    # Load tid mappings
    print("Loading TID mappings...")
    tid2item_id, reverse_mapping, word_to_keys, num_keywords = load_tid_mappings(dataset_name)
    print(f"Detected {num_keywords} keywords per item")
    
    # Load pre-generated outputs
    print(f"Loading generation outputs from {gen_output_file}...")
    gen_outputs = load_generation_outputs(gen_output_file)
    print(f"Loaded {len(gen_outputs)} generation outputs")
    
    # Load item frequency
    print(f"Loading item frequency from {item_freq_file}...")
    item_freq = load_item_frequency(item_freq_file)
    print(f"Loaded frequency for {len(item_freq)} items")
    
    # Process generation outputs with top-2 frequency sampling
    print("Processing outputs with top-2 frequency sampling...")
    results = process_generation_outputs(
        gen_outputs, tid2item_id, reverse_mapping, word_to_keys, item_freq,
        rerank_by_popularity=rerank_by_popularity
    )
    print(f"Processed {len(results)} results")
    
    # Calculate metricsprocess_generation_outputs
    print("Calculating metrics...")
    recall_metrics, scores, dcg = calculate_recall(results)
    
    # Output results
    print("\n" + "="*50)
    print("Evaluation Results (Top-2 Frequency Sampling)")
    print("="*50)
    for metric, value in recall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save detailed results
    output_file = gen_output_file.replace('.json', '_eval_top2.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': recall_metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {output_file}")
    
    return recall_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="beauty")
    parser.add_argument("--rerank_by_popularity", action="store_true",
                        help="Re-rank the top-20 candidates by item popularity")
    args = parser.parse_args()

    dataset_name = args.dataset
    # gen_output_file = f"./results/{dataset_name}_generation_outputs_new_eval_cop7.json"
    gen_output_file = f"./results/{dataset_name}_generation_outputs_tid.json"
    item_freq_file = f"../data/{dataset_name}_item_freq.pkl"

    main(gen_output_file, item_freq_file, dataset_name, rerank_by_popularity=False)