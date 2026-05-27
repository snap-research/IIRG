import json
import re
from tqdm import tqdm
from collections import Counter, defaultdict


# Will be populated at runtime based on frequency threshold
YELP_NOISY_CATEGORIES = set()

def build_yelp_noisy_categories(items, threshold=0.05):
    """Build set of noisy categories that appear in >threshold fraction of items."""
    from collections import Counter
    cat_counter = Counter()
    for v in items.values():
        title = v.get('title', '')
        m = re.search(r'Category:\s*(.+?),\s*Name:', title)
        if m:
            cats = [c.strip() for c in m.group(1).split(',')]
            for c in cats:
                if c:
                    cat_counter[c] += 1
    
    n_items = len(items)
    noisy = {cat for cat, cnt in cat_counter.items() if cnt / n_items > threshold}
    print(f"Filtered {len(noisy)} noisy categories (>{threshold*100:.0f}% of items):")
    for cat in sorted(noisy, key=lambda x: -cat_counter[x]):
        print(f"  {cat}: {cat_counter[cat]} ({cat_counter[cat]/n_items*100:.1f}%)")
    return noisy


def clean_yelp_title(title):
    """Remove formatting labels from Yelp title, filter generic categories,
    and combine all text into one clean string.
    
    Example input:
        "Key information location: OH, Cleveland, 1541 E 38th St, Ste 101 | Sub information: (Category: Restaurants, Vietnamese, Soup,  Name: Pho Lee's Vietnamese Restaurant)."
    Example output:
        "Cleveland, OH, Pho Lee's Vietnamese Restaurant, Vietnamese, Soup"
    """
    # Extract components
    loc_match = re.search(r'Key information location:\s*([^|]+)', title)
    cat_match = re.search(r'Category:\s*(.+?),\s*Name:', title)
    name_match = re.search(r'Name:\s*(.+?)\)', title)
    
    parts = []
    
    # Location first: city and state only (skip street address)
    if loc_match:
        loc_parts = [p.strip() for p in loc_match.group(1).split(',')]
        # First part is state, second is city
        if len(loc_parts) >= 2:
            parts.append(loc_parts[1])  # city
            parts.append(loc_parts[0])  # state
    
    # Name
    if name_match:
        parts.append(name_match.group(1).strip())
    
    # Filter categories: keep only specific ones, max 5
    if cat_match:
        cats = [c.strip() for c in cat_match.group(1).split(',')]
        cats = [c for c in cats if c and c not in YELP_NOISY_CATEGORIES]
        parts.extend(cats[:3])
    
    return ', '.join(parts)

def create_simple_mapping(dataset, max_words=None, auto_detect=False):
    """Create simple parent_asin mapping
    
    Args:
        dataset: Dataset name
        max_words: Maximum number of words to keep. If None, keep all words.
                   If specified, truncate to first max_words items.
        auto_detect: If True, automatically use the most common keyword count.
                     Overrides max_words if both are specified.
    """
    input_file = f"./data/{dataset}_tid_updated.jsonl"
    output_file = f"./data/{dataset}_id2meta_text.json"
    
    all_items = []
    parent_asin2meta = {}
    word_length_stats = Counter()  # Track word length distribution
    
    # First pass: collect statistics
    print("Analyzing keyword distribution...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                word_count = len(item["summary_words"])
                word_length_stats[word_count] += 1
    
    # Print word length statistics
    print("\nWord length distribution:")
    for length in sorted(word_length_stats.keys()):
        print(f"  {length} words: {word_length_stats[length]} items")
    
    # Determine max_words if auto_detect is enabled
    if auto_detect:
        most_common_length = word_length_stats.most_common(1)[0][0]
        print(f"\n✓ Auto-detected most common keyword count: {most_common_length}")
        max_words = most_common_length
    
    # Second pass: process and save items
    print("\nProcessing items...")
    
    # For Yelp: build noisy category set from item data
    if dataset == "yelp":
        global YELP_NOISY_CATEGORIES
        item_file = f"../data/{dataset}.item.json"
        with open(item_file, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
        YELP_NOISY_CATEGORIES = build_yelp_noisy_categories(item_data, threshold=0.15)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing data"):
            if line.strip():
                item = json.loads(line)
                
                # Clean Yelp title format
                if dataset == "yelp" and "title" in item:
                    item["title"] = clean_yelp_title(item["title"])
                
                # Optionally truncate to max_words
                if max_words is not None and len(item["summary_words"]) > max_words:
                    item["summary_words"] = item["summary_words"][:max_words]
                
                item["summary_words"] = ["-".join(word.split()) for word in item["summary_words"]]
                all_items.append(item)
                parent_asin = item.get('id')
                if parent_asin:
                    parent_asin2meta[parent_asin] = item
    
    # Save mapping
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parent_asin2meta, f, ensure_ascii=False, indent=2)
    
    print(f"\nCompleted! Processed {len(parent_asin2meta)} products")
    if max_words:
        print(f"Using {max_words} keywords per item")
    return parent_asin2meta

# Example usage function
def query_product_info(parent_asin2meta, asin):
    """Query product information for a specific ASIN"""
    if asin in parent_asin2meta:
        meta = parent_asin2meta[asin]
        return meta
    else:
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yelp", help="Dataset name (e.g., sports, toys, beauty, yelp)")
    args = parser.parse_args()

    # Options:
    # - auto_detect=True: Use the most common keyword count (recommended)
    # - max_words=5: Manually set to 5 keywords
    # - max_words=None, auto_detect=False: Keep all keywords as-is
    
    # Create mapping with auto-detection
    mapping = create_simple_mapping(args.dataset, auto_detect=True)
    
    # Example query
    sample_asin = list(mapping.keys())[0] if mapping else None
    if sample_asin:
        info = query_product_info(mapping, sample_asin)
        print(f"\nExample query - ASIN: {sample_asin}")
        print(f"Product information: {info}")