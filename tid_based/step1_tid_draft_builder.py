import os
import json
import random
import re
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list


def load_similarities(similarity_file):
    with open(similarity_file, 'r') as f:
        similarities = json.load(f)
    return similarities


def select_exact_neighbors(similarities_dict, K2=10, K3=50, K4=500, N=3):
    """
    Select neighbors from specific ranges:
    - k1: Top-1 item
    - k2: Sample N items from ranks 3 to K2
    - k3: Sample N items from ranks (K2+1) to K3
    - k4: Sample N items from ranks (K3+1) to K4
    
    Args:
        similarities_dict: dict mapping item_id to list of neighbor IDs (already sorted by similarity)
        K2: upper bound for k2 range (3 to K2)
        K3: upper bound for k3 range (K2+1 to K3)
        K4: upper bound for k4 range (K3+1 to K4)
        N: number of items to sample from each range
    """
    selected = {}
    for item_id, nbs in similarities_dict.items():
        n = len(nbs)
        
        # k1: Top-1 item (index 0)
        k1 = [nbs[0]] if n > 0 else []
        
        # k2: ranks 3 to K2 (indices 2 to K2-1)
        candidates_k2 = [nbs[i] for i in range(2, min(K2, n))]
        k2 = random.sample(candidates_k2, min(N, len(candidates_k2)))
        
        # k3: ranks K2+1 to K3 (indices K2 to K3-1)
        candidates_k3 = [nbs[i] for i in range(K2, min(K3, n))]
        k3 = random.sample(candidates_k3, min(N, len(candidates_k3)))
        
        # k4: ranks K3+1 to K4 (indices K3 to K4-1)
        candidates_k4 = [nbs[i] for i in range(K3, min(K4, n))]
        k4 = random.sample(candidates_k4, min(N, len(candidates_k4)))
        
        selected[item_id] = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k5": []}
    return selected

def remove_highest_hierarchy(text) : 
    parts = text.split(">", 1)
    result = parts[1].strip() if len(parts) > 1 else text
    return result

n_descrip = 50

def build_taxonomy_prompt(anchor_item, top3, top10, top30, top100, top300, meta_data, domain_type):
    anchor = meta_data.get(str(anchor_item), {})
    target_node_descriptions = (
        f"Title: [{anchor.get('title', '')}], "
        f"Category: [{remove_highest_hierarchy(anchor.get('categories', ''))}],"
        f"Description: [{anchor.get('description', '')[:n_descrip]}]"
    )
    top_3_desc = ""
    top_10_desc = ""
    top_30_desc = ""
    top_100_desc = ""
    top_300_desc = ""

    for v in top3:
        m = meta_data.get(str(v), {})
        top_3_desc += f"- <Title: [{m.get('title', '')}], Category: [{remove_highest_hierarchy(m.get('categories', ''))}], Description: [{m.get('description', '')[:n_descrip]}]>\n"

    for v in top10:
        m = meta_data.get(str(v), {})
        top_10_desc += f"- <Title: [{m.get('title', '')}], Category: [{remove_highest_hierarchy(m.get('categories', ''))}], Description: [{m.get('description', '')[:n_descrip]}]>\n"

    for v in top30:
        m = meta_data.get(str(v), {})
        top_30_desc += f"- <Title: [{m.get('title', '')}], Category: [{remove_highest_hierarchy(m.get('categories', ''))}], Description: [{m.get('description', '')[:n_descrip]}]>\n"

    for v in top100:
        m = meta_data.get(str(v), {})
        top_100_desc += f"- <Title: [{m.get('title', '')}], Category: [{remove_highest_hierarchy(m.get('categories', ''))}], Description: [{m.get('description', '')[:n_descrip]}]>\n"

    return f"""
You are an item categorizing assistant.

All items are from the {domain_type} domain, and your taxonomy should be more specific than this domain.

Given one anchor item and four nested similarity sets from the {domain_type} domain, generate a 4-level taxonomy for the anchor item. Attributes should be more specific than {domain_type}.
You must not use {domain_type} alone as an attribute.

Task:
For each level, identify the single shared characteristic that ALL items in that group (including the anchor) have in common. Each level should capture what unifies the group, not what describes any individual item.

- Level 1: Broad characteristic the anchor and ALL Top-500 items share (broadest common trait)
- Level 2: Narrower characteristic the anchor and ALL Top-50 items share (more specific than Level 1)
- Level 3: Specific characteristic the anchor and ALL Top-5 items share (more specific than Level 2)
- Level 4: Distinctive characteristic of the anchor item and Group 4 that is NOT shared by any items in Groups 1, 2, or 3 (what makes the anchor unique)

Constraints:
- Different levels must not be the same.
- NEVER use "{domain_type}" alone.
- Level 1 must be more specific than the given domain: "{domain_type}".
- Level 4 must be highly specific.
- The four levels must be different and non-redundant.
- Level 1 must be a single word.
- Level 2 can be up to 2 words.
- Level 3 can be up to 2 words.
- Level 4 can be up to 2 words.
- Do not use plural.
- Exactly 4 output lines only
- Levels 1, 2, and 3 must describe what is SHARED across the group, without excluding any item

Return exactly:
Level 1: ...
Level 2: ...
Level 3: ...
Level 4: ...

Domain:
{domain_type}

Anchor item:
{target_node_descriptions}

Group 1 (find the broadest shared trait)::
{top_100_desc}

Group 2 (find a narrower shared trait):
{top_30_desc}

Group 3 (find a specific shared trait):
{top_10_desc}

Group 4 (find what makes the anchor unique):
{top_3_desc}

Notes:
- Different levels should not be redundant (e.g., [doll, doll, ...] is not allowed).
- Do not use synonyms for different levels (e.g., [guns, rifles, ...] is not allowed).
- NEVER USE "{domain_type}" alone as a level.

Output:
""".strip()


def parse_taxonomy_output(content):
    """Parse model output into a list of 4 level labels."""
    words = []
    for level_num in range(1, 5):
        pattern = rf"Level\s*{level_num}\s*:\s*(.+)"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            label = match.group(1).strip().strip('\"\'\'').lower()
            label = re.sub(r'\s+', '-', label)
            words.append(label)
        else:
            words.append("")
    while len(words) < 4:
        words.append("")
    return words[:4]


def process_batch_on_gpu(rank, data_slice, output_queue, model_name,
                         sampled, meta_data, domain_type, batch_size=2):
    print(f"Rank {rank}: Initializing model...")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True
    )
    model.eval()

    print(f"Rank {rank}: Starting to process {len(data_slice)} items")

    results = []

    for i in tqdm(range(0, len(data_slice), batch_size), desc=f"Rank {rank}"):
        batch_items = data_slice[i:i + batch_size]
        batch_results = process_single_batch(
            batch_items, model, tokenizer, device, sampled, meta_data, domain_type
        )
        results.extend(batch_results)

    output_queue.put((rank, results))
    print(f"Rank {rank}: Processing completed with {len(results)} items")


def process_single_batch(items, model, tokenizer, device, sampled, meta_data, domain_type):
    prompts = []
    for item in items:
        item_id = item['id']

        groups = sampled.get(item_id, {"k1": [], "k2": [], "k3": [], "k4": [], "k5": []})
        prompt = build_taxonomy_prompt(
            anchor_item=item_id,
            top3=groups["k1"],
            top10=groups["k2"],
            top30=groups["k3"],
            top100=groups["k4"],
            top300=groups["k5"],
            meta_data=meta_data,
            domain_type=domain_type,
        )

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(text)

    tokenizer.padding_side = 'left'
    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
        max_length=32768
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

    results = []
    for i, (item, input_ids, output_ids) in enumerate(
        zip(items, model_inputs.input_ids, generated_ids)
    ):
        generated_output_ids = output_ids[len(input_ids):].tolist()
        content = tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip("\n")

        item_id = item['id']
        words = parse_taxonomy_output(content)

        groups = sampled.get(item_id, {"k1": [], "k2": [], "k3": []})

        item_copy = item.copy()
        item_copy['llm_output'] = content
        item_copy['summary_words'] = words
        item_copy['similar_item_ids'] = groups["k1"]

        results.append(item_copy)

    return results


def analyze_statistics(all_items):
    print("\n" + "=" * 50)
    print("Statistical Analysis Results")
    print("=" * 50)

    all_words = []
    word_freq = Counter()
    word_by_position = [Counter() for _ in range(4)]

    for item in all_items:
        words = item.get('summary_words', [])
        all_words.extend([word for word in words if word])
        for i, word in enumerate(words):
            if i < 4 and word:
                word_by_position[i][word] += 1

    word_freq.update(all_words)

    print(f"\n1. Overall Vocabulary Statistics:")
    print(f"   Total words: {len(all_words)}")
    print(f"   Unique words: {len(word_freq)}")
    print(f"   Top 20 most frequent words:")
    for word, count in word_freq.most_common(20):
        print(f"     {word}: {count}")

    print(f"\n2. Vocabulary Statistics by Level:")
    positions = ['Level 1 (Broad)', 'Level 2 (Narrower)', 'Level 3 (Specific)', 'Level 4 (Unique)']
    for i, (pos, counter) in enumerate(zip(positions, word_by_position)):
        print(f"   Top 10 words for {pos}:")
        for word, count in counter.most_common(10):
            print(f"     {word}: {count}")

    print(f"\n3. Conflict Analysis:")
    summary_tuples = [tuple(item.get('summary_words', [])) for item in all_items]
    tuple_counter = Counter(summary_tuples)

    duplicate_tuples = [(tup, count) for tup, count in tuple_counter.items() if count > 1]
    total_conflicts = sum(count - 1 for tup, count in duplicate_tuples)
    conflict_rate = total_conflicts / len(all_items) if all_items else 0

    print(f"   Identical summaries count: {len(duplicate_tuples)}")
    print(f"   Conflicting items count: {total_conflicts}")
    print(f"   Conflict rate: {conflict_rate:.4f}")

    if duplicate_tuples:
        print(f"   Top 10 most frequent conflicts:")
        for tup, count in sorted(duplicate_tuples, key=lambda x: x[1], reverse=True)[:10]:
            print(f"     {tup}: appears {count} times")

    print(f"\n4. Validity Check:")
    valid_items = 0
    partial_items = 0

    for item in all_items:
        words = item.get('summary_words', [])
        valid_words = [word for word in words if word]
        if len(valid_words) == 4:
            valid_items += 1
        elif len(valid_words) > 0:
            partial_items += 1

    invalid_items = len(all_items) - valid_items - partial_items

    print(f"   Complete 4-level items: {valid_items}/{len(all_items)} ({valid_items / len(all_items) * 100:.2f}%)")
    print(f"   Partially valid items: {partial_items}/{len(all_items)} ({partial_items / len(all_items) * 100:.2f}%)")
    print(f"   Invalid items: {invalid_items}/{len(all_items)} ({invalid_items / len(all_items) * 100:.2f}%)")

    return {
        'word_frequency': dict(word_freq.most_common()),
        'position_frequency': [dict(counter.most_common()) for counter in word_by_position],
        'conflict_analysis': {
            'total_conflicts': total_conflicts,
            'conflict_rate': conflict_rate,
            'duplicate_tuples': [(list(tup), count) for tup, count in duplicate_tuples]
        },
        'validity_analysis': {
            'valid_items': valid_items,
            'partial_items': partial_items,
            'invalid_items': invalid_items
        },
    }


DOMAIN_MAP = {
    "sports": "Sports & Outdoors",
    "toys": "Toys",
    "beauty": "Beauty",
}

def main(dataset, sample_fraction=1.0):
    
    domain_type = DOMAIN_MAP.get(dataset, dataset.capitalize())
    model_name = "Qwen/Qwen3.5-9B"
    
    gpu_ids = list(range(torch.cuda.device_count()))

    num_gpus = len(gpu_ids)

    data_file = f"../data/{dataset}.item.json"
    dataset_similarities_file = f"../data/{dataset}_semantic_neighbors.json"

    print(f"Loading data: {data_file}")
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")

    if sample_fraction < 1.0:
        original_count = len(data)
        sample_size = max(1, int(len(data) * sample_fraction))
        data = random.sample(data, sample_size)
        print(f"Sampling {sample_fraction * 100:.1f}% of data: {len(data)} items (from {original_count} total)")

    print(f"Loading similarity data: {dataset_similarities_file}")
    similarities_dict = load_similarities(dataset_similarities_file)
    print(f"Loaded similarity information for {len(similarities_dict)} items")

    # Build meta_data dict
    meta_data = {}
    for d in data:
        meta_data[d["id"]] = {
            "title": d.get("title", ""),
            "categories": d.get("categories", ""),
            "description": d.get("description", ""),
        }
    # Also add meta for neighbor items that may not be in the sampled data
    all_data = load_data(data_file)
    for d in all_data:
        if d["id"] not in meta_data:
            meta_data[d["id"]] = {
                "title": d.get("title", ""),
                "categories": d.get("categories", ""),
                "description": d.get("description", ""),
            }

    # Select neighbors with window-based approach
    K1 = 1
    # Select neighbors from ranges
    K2 = 10   # k2: ranks 5 to K2
    K3 = 50   # k3: ranks K2+1 to K3
    K4 = 500  # k4: ranks K3+1 to K4
    N = 3     # number of items to sample from each range
    print(f"Selecting neighbors (K2={K2}, K3={K3}, K4={K4}, N={N})...")
    sampled = select_exact_neighbors(similarities_dict, K2=K2, K3=K3, K4=K4, N=N)
    print(f"Selected neighbors for {len(sampled)} items")

    # Print first prompt for debugging
    first_item = data[0]
    first_groups = sampled.get(first_item['id'], {"k1": [], "k2": [], "k3": [], "k4": [], "k5": []})
    first_prompt = build_taxonomy_prompt(
        anchor_item=first_item['id'],
        top3=first_groups["k1"],
        top10=first_groups["k2"],
        top30=first_groups["k3"],
        top100=first_groups["k4"],
        top300=first_groups["k5"],
        meta_data=meta_data,
        domain_type=domain_type,
    )
    print("\n" + "=" * 50)
    print("FIRST PROMPT:")
    print("=" * 50)
    print(first_prompt)
    print("=" * 50 + "\n")

    chunk_size = len(data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + chunk_size
        data_chunks.append(data[start_idx:end_idx])

    print(f"Dataset split into {num_gpus} chunks, chunk sizes: {[len(chunk) for chunk in data_chunks]}")

    processes = []
    output_queue = mp.Queue()

    print("Starting multi-process processing...")
    start_time = time.time()

    for i in range(num_gpus):
        p = mp.Process(
            target=process_batch_on_gpu,
            args=(gpu_ids[i], data_chunks[i], output_queue, model_name,
                  sampled, meta_data, domain_type, 16)
        )
        processes.append(p)
        p.start()

    all_results = []
    for _ in range(num_gpus):
        rank, results = output_queue.get()
        print(f"Received {len(results)} results from Rank {rank}")
        all_results.extend(results)

    for p in processes:
        p.join()

    end_time = time.time()
    print(f"Multi-process processing completed, total time: {end_time - start_time:.2f} seconds")

    # Save results
    output_file = f"./data/{dataset}_tid_draft.jsonl"
    print(f"Saving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Statistical analysis
    stats = analyze_statistics(all_results)

    # Save statistics
    stats_file = f"./data/{dataset}_tid_draft_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_file}")

    # Output examples
    print(f"\nExample outputs (first 3 items):")
    for i, item in enumerate(all_results[:5]):
        print(f"\nItem ID: {item['id']}")
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"Categories: {item.get('categories', 'N/A')}")
        print(f"Summary words: {item.get('summary_words', [])}")
        print(f"LLM output: {item.get('llm_output', '')}")

    print(f"\nProcessing completed! Total items processed: {len(all_results)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys", help="Dataset name (e.g., sports, toys, beauty, yelp)")
    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    main(args.dataset, sample_fraction=1.0)