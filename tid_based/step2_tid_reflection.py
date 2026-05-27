"""
Hierarchical Taxonomy Refinement Pipeline with Multi-GPU Support

This version distributes cluster canonicalization across multiple GPUs:
1. Main process: loads data, clusters, encodes
2. Worker processes: each canonicalizes assigned clusters on a separate GPU
3. Main process: merges results, updates dataset, checks convergence

Based on step2_TID_reflection.py but with multi-GPU parallelization.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class TaxonomyRefiner:
    """Single-GPU version of TaxonomyRefiner (used by worker processes)."""

    DOMAIN_CONTEXT = {
        "sports": "Sports & Outdoors",
        "toys": "Toys",
        "beauty": "Beauty",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        rank: int = 0,
        max_prompt_terms: int = 64,
        max_input_length: int = 8192,
        enforce_parent_distinct: bool = True,
        data_name: str = "toys",
    ):
        self.rank = rank
        self.max_prompt_terms = max_prompt_terms
        self.max_input_length = max_input_length
        self.enforce_parent_distinct = enforce_parent_distinct
        self.data_name = data_name

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            llm_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            llm_dtype = torch.float32

        print(f"[Rank {rank}] Using device: {self.device}")

        # LLM only (embedding loaded separately)
        print(f"[Rank {rank}] Loading LLM: {model_name}")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        llm_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": llm_dtype,
        }
        if torch.cuda.is_available():
            llm_kwargs["device_map"] = f"cuda:{rank}"
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            **llm_kwargs,
        )
        if not torch.cuda.is_available():
            self.llm.to(self.device)
        self.llm.eval()

        print(f"[Rank {rank}] LLM loaded successfully")

    @staticmethod
    def clean_tag(tag: Optional[str]) -> str:
        if tag is None:
            return ""
        tag = str(tag).strip()
        tag = re.sub(r"\s+", " ", tag)
        return tag

    @staticmethod
    def normalize_for_compare(tag: str) -> str:
        tag = tag.strip().lower()
        tag = re.sub(r"\s+", " ", tag)
        return tag

    def build_canonicalization_prompt(
        self,
        cluster_keywords: List[str],
        parent_tags: Optional[List[str]] = None,
    ) -> str:
        terms = [self.clean_tag(t) for t in cluster_keywords if self.clean_tag(t)]
        terms = terms[: self.max_prompt_terms]
        terms_str = "[" + ", ".join(json.dumps(t, ensure_ascii=False) for t in terms) + "]"

        if parent_tags:
            parent_tags = [self.clean_tag(t) for t in parent_tags if self.clean_tag(t)]
            parent_desc = (
                "\nParent categories (your output must be more specific than these and distinct from these): "
                + ", ".join(parent_tags[:5])
            )
            specificity_rule = (
                "\n9. Every output term must be more specific than and distinct from the parent categories."
                "\n10. Do not collapse a child term into a parent category name."
            )
        else:
            parent_desc = ""
            specificity_rule = ""

        domain_desc = ""
        if self.data_name in self.DOMAIN_CONTEXT:
            domain_name = self.DOMAIN_CONTEXT[self.data_name]
            domain_desc = f"""
Context: These tags come from the "{domain_name}" product domain.
IMPORTANT: Do NOT use "{self.data_name}" itself as an output tag. The resulting tags should be more fine-grained and specific product characteristics or subcategories within {domain_name}. For example, instead of outputting "{self.data_name}", output more specific terms like specific sport types, product features, or categories."""

        return f"""
You are a taxonomy normalization function.

Your task is to normalize a list of terms into canonical forms.{domain_desc}

Rules:
1. Convert plural terms to singular.
2. Merge synonyms into one canonical term.
3. Keep only semantically meaningful distinctions.
4. Use short, consistent, singular canonical terms.
5. Do not introduce concepts that are not supported by the input.
6. Use hyphens for multi-word concepts when that improves consistency (example: "ice cream" -> "ice-cream").
7. Do not use the dataset name or broad domain tags as output terms; instead, provide more specific characteristic terms.{specificity_rule}


Output requirements:
- Return valid JSON only.
- Do not include markdown fences.
- Do not include any explanation outside the JSON.
- Return exactly one JSON object mapping every input term to a canonical term.
- Every input term must appear exactly once as a key.
- The values must be strings.

Format:
{{
  "original_term_1": "canonical_term_1",
  "original_term_2": "canonical_term_2",
  ...
}}

Input terms:
{terms_str}{parent_desc}
""".strip()

    def apply_chat_template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def extract_json_dict(text: str) -> Dict[str, str]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text[start:])
        if not isinstance(obj, dict):
            raise json.JSONDecodeError("Top-level JSON is not a dict", text, start)

        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, str):
                out[k] = v
        return out

    def canonicalize_cluster(
        self,
        cluster_keywords: List[str],
        parent_tags: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
    ) -> Tuple[Dict[str, str], Dict]:
        cluster_keywords = [self.clean_tag(x) for x in cluster_keywords if self.clean_tag(x)]
        if not cluster_keywords:
            return {}, {"status": "empty_cluster"}

        if len(cluster_keywords) == 1:
            kw = cluster_keywords[0]
            return {kw: kw}, {
                "status": "singleton",
                "cluster_keywords": cluster_keywords,
                "parent_tags": parent_tags or [],
                "raw_output": None,
                "parsed_mapping": {kw: kw},
            }

        prompt = self.build_canonicalization_prompt(cluster_keywords, parent_tags=parent_tags)
        chat_text = self.apply_chat_template(prompt)

        model_inputs = self.llm_tokenizer(
            [chat_text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.llm.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )

        generated_output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output_text = self.llm_tokenizer.decode(
            generated_output_ids,
            skip_special_tokens=True,
        ).strip()

        debug_info = {
            "status": "ok",
            "cluster_keywords": cluster_keywords,
            "parent_tags": parent_tags or [],
            "raw_output": output_text,
            "parsed_mapping": None,
            "parse_error": None,
            "final_mapping": None,
        }

        try:
            raw_mapping = self.extract_json_dict(output_text)
            debug_info["parsed_mapping"] = raw_mapping
        except Exception as e:
            identity = {kw: kw for kw in cluster_keywords}
            debug_info["status"] = "parse_failed_identity_fallback"
            debug_info["parse_error"] = repr(e)
            debug_info["final_mapping"] = identity
            print(f"    [WARN] JSON parse failed for cluster ({len(cluster_keywords)} terms): {repr(e)}")
            return identity, debug_info

        parent_norm = set()
        if parent_tags:
            parent_norm = {self.normalize_for_compare(t) for t in parent_tags if self.clean_tag(t)}

        final_mapping = {}
        cluster_set = set(cluster_keywords)

        for kw in cluster_keywords:
            val = raw_mapping.get(kw, kw)
            val = self.clean_tag(val) or kw

            if self.enforce_parent_distinct and parent_norm:
                if self.normalize_for_compare(val) in parent_norm:
                    val = kw

            final_mapping[kw] = val

        final_mapping = {k: v for k, v in final_mapping.items() if k in cluster_set}
        debug_info["final_mapping"] = final_mapping
        return final_mapping, debug_info


def persistent_worker(
    rank: int,
    gpu_id: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    model_name: str,
    embedding_model_name: str,
    max_prompt_terms: int,
    max_input_length: int,
    enforce_parent_distinct: bool,
    data_name: str,
):
    """
    Persistent worker process: loads model once, then loops on work from input_queue.

    Protocol:
        - input_queue receives (cluster_batch,) or None to signal shutdown.
        - output_queue sends (rank, results) or (rank, "ERROR: ...").
    """
    print(f"[Rank {rank} / GPU {gpu_id}] Starting persistent worker...")

    try:
        refiner = TaxonomyRefiner(
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            rank=gpu_id,
            max_prompt_terms=max_prompt_terms,
            max_input_length=max_input_length,
            enforce_parent_distinct=enforce_parent_distinct,
            data_name=data_name,
        )
    except Exception as e:
        print(f"[Rank {rank} / GPU {gpu_id}] ERROR loading model: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Drain the queue so the main process doesn't hang
        while True:
            msg = input_queue.get()
            if msg is None:
                break
            output_queue.put((rank, f"ERROR: {type(e).__name__}: {str(e)}"))
        return

    print(f"[Rank {rank} / GPU {gpu_id}] Model loaded, waiting for work...")

    while True:
        msg = input_queue.get()
        if msg is None:
            print(f"[Rank {rank} / GPU {gpu_id}] Received shutdown signal.")
            break

        cluster_batch = msg
        try:
            results = []
            for cluster_id, cluster_terms, parent_tags in cluster_batch:
                cluster_mapping, cluster_debug = refiner.canonicalize_cluster(
                    cluster_terms,
                    parent_tags=parent_tags,
                )
                cluster_debug["cluster_id"] = cluster_id
                cluster_debug["cluster_size"] = len(cluster_terms)
                results.append((cluster_mapping, cluster_debug))

            print(f"[Rank {rank} / GPU {gpu_id}] Processed {len(results)} clusters")
            output_queue.put((rank, results))
        except Exception as e:
            print(f"[Rank {rank} / GPU {gpu_id}] ERROR processing: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            output_queue.put((rank, f"ERROR: {type(e).__name__}: {str(e)}"))


class TaxonomyRefinerMultiGPU:
    """Main orchestrator for multi-GPU taxonomy refinement."""

    DOMAIN_CONTEXT = {
        "sports": "Sports & Outdoors",
        "toys": "Toys",
        "beauty": "Beauty",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        gpu_ids: List[int] = None,
        max_prompt_terms: int = 64,
        max_input_length: int = 8192,
        enforce_parent_distinct: bool = True,
        data_name: str = "toys",
    ):
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.gpu_ids = gpu_ids or [0, 1, 2, 3, 4, 5, 6, 7]
        self.num_gpus = len(self.gpu_ids)
        self.max_prompt_terms = max_prompt_terms
        self.max_input_length = max_input_length
        self.enforce_parent_distinct = enforce_parent_distinct
        self.data_name = data_name

        print(f"Using {self.num_gpus} GPUs: {self.gpu_ids}")

        # Persistent worker state
        self._workers = []       # list of (rank, process)
        self._input_queues = []  # one per worker
        self._output_queue = None

        # Load embedding model once (used by main process)
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.embedding_model.eval()
        if torch.cuda.is_available():
            self.embedding_model.to(f"cuda:{self.gpu_ids[0]}")
        print("Embedding model loaded")

    # ---- Persistent worker lifecycle ----

    def start_workers(self):
        """Spawn persistent workers (one per GPU). Call once before any stabilization."""
        if self._workers:
            return  # already running

        self._output_queue = mp.Queue()
        self._input_queues = []
        self._workers = []

        for rank, gpu_id in enumerate(self.gpu_ids):
            iq = mp.Queue()
            p = mp.Process(
                target=persistent_worker,
                args=(
                    rank,
                    gpu_id,
                    iq,
                    self._output_queue,
                    self.model_name,
                    self.embedding_model_name,
                    self.max_prompt_terms,
                    self.max_input_length,
                    self.enforce_parent_distinct,
                    self.data_name,
                ),
            )
            p.start()
            self._input_queues.append(iq)
            self._workers.append((rank, p))

        print(f"Started {len(self._workers)} persistent workers.")

    def stop_workers(self):
        """Send shutdown signal and join all workers."""
        for iq in self._input_queues:
            iq.put(None)
        for rank, p in self._workers:
            p.join(timeout=60)
            if p.is_alive():
                print(f"[Rank {rank}] Worker did not exit, terminating...")
                p.terminate()
        self._workers = []
        self._input_queues = []
        self._output_queue = None
        print("All workers stopped.")

    def last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
        if left_padding:
            return last_hidden_states[:, -1]
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            seq_lens,
        ]

    def tokenize_with_eos(self, texts: List[str], max_length: int = 64):
        batch = self.embedding_tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length - 1,
        )

        eos_id = self.embedding_tokenizer.eos_token_id
        if eos_id is not None:
            for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
                input_ids.append(eos_id)
                attention_mask.append(1)

        return self.embedding_tokenizer.pad(batch, padding=True, return_tensors="pt")

    @torch.no_grad()
    def encode_texts(self, texts: List[str], max_length: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        batch_dict = self.tokenize_with_eos(texts, max_length=max_length)
        device = next(self.embedding_model.parameters()).device
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        outputs = self.embedding_model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().numpy()

    def encode_batch(self, texts: List[str], batch_size: int = 32, max_length: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.encode_texts(batch, max_length=max_length)
            embeddings_list.append(batch_embeddings)
        return np.vstack(embeddings_list)

    def find_top_k_closest_tags(
        self,
        cluster_centroid: np.ndarray,
        parent_embeddings: np.ndarray,
        parent_tags: List[str],
        k: int = 5,
    ) -> List[str]:
        if len(parent_tags) == 0 or parent_embeddings.size == 0:
            return []

        centroid = cluster_centroid.astype(np.float32)
        denom = np.linalg.norm(centroid)
        if denom > 0:
            centroid = centroid / denom

        sims = parent_embeddings @ centroid
        top_k_indices = np.argsort(-sims)[: min(k, len(parent_tags))]
        return [parent_tags[i] for i in top_k_indices]

    def cluster_keywords_balanced(
        self,
        keywords: List[str],
        embeddings: np.ndarray,
        n_clusters: int,
        max_iterations: int = 10,
    ) -> Tuple[Dict[int, List[str]], np.ndarray]:
        """Balanced KMeans clustering."""
        if not keywords:
            return {}, np.zeros((0, 0), dtype=np.float32)

        n_clusters = max(1, min(n_clusters, len(keywords)))

        if n_clusters == 1:
            return {0: keywords[:]}, embeddings.mean(axis=0, keepdims=True)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(embeddings)

        assignments = [[] for _ in range(n_clusters)]
        for keyword_idx, cluster_id in enumerate(cluster_labels):
            assignments[cluster_id].append(keyword_idx)

        target_low = len(keywords) // n_clusters
        target_high = math.ceil(len(keywords) / n_clusters)

        for _ in range(max_iterations):
            sizes = [len(x) for x in assignments]
            underfull = [i for i, s in enumerate(sizes) if s < target_low]
            overfull = [i for i, s in enumerate(sizes) if s > target_high]

            if not underfull or not overfull:
                break

            for small_id in underfull:
                while len(assignments[small_id]) < target_low:
                    overfull = [i for i, s in enumerate([len(x) for x in assignments]) if s > target_high]
                    if not overfull:
                        break

                    large_id = max(overfull, key=lambda cid: len(assignments[cid]))
                    if not assignments[large_id]:
                        break

                    small_centroid = np.mean([embeddings[idx] for idx in assignments[small_id]], axis=0) if assignments[small_id] else kmeans.cluster_centers_[small_id]

                    large_points = assignments[large_id]
                    distances = [
                        np.linalg.norm(embeddings[idx] - small_centroid)
                        for idx in large_points
                    ]
                    move_idx = large_points[int(np.argmin(distances))]

                    assignments[large_id].remove(move_idx)
                    assignments[small_id].append(move_idx)

        centroids = []
        clusters: Dict[int, List[str]] = {}
        for cluster_id, indices in enumerate(assignments):
            clusters[cluster_id] = [keywords[idx] for idx in indices]
            centroid = np.mean([embeddings[idx] for idx in indices], axis=0) if indices else kmeans.cluster_centers_[cluster_id]
            centroids.append(centroid)

        return clusters, np.vstack(centroids)

    def merge_mappings(
        self,
        mapping_list: List[Dict[str, str]],
        universe_terms: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Merge multiple mappings with majority vote."""
        votes: Dict[str, Counter] = defaultdict(Counter)

        for mapping in mapping_list:
            for orig, canonical in mapping.items():
                orig = orig.strip()
                canonical = canonical.strip()
                if not orig:
                    continue
                votes[orig][canonical or orig] += 1

        merged = {}
        for orig, counter in votes.items():
            merged[orig] = counter.most_common(1)[0][0]

        if universe_terms is not None:
            for term in universe_terms:
                term = term.strip()
                if term and term not in merged:
                    merged[term] = term

        return merged

    def stabilize_level_multiprocess(
        self,
        level_idx: int,
        data: List[dict],
        parent_canonical_tags: Optional[List[str]] = None,
        parent_embeddings: Optional[np.ndarray] = None,
        threshold: int = 10,
        max_iterations: int = 5,
        batch_size: int = 32,
        cluster_size_target: int = 30,
    ) -> Tuple[Dict[str, str], List[str], np.ndarray, Dict[str, int]]:
        """Stabilize a taxonomy level using multi-GPU cluster canonicalization."""
        print("\n" + "=" * 72)
        print(f"Stabilizing level {level_idx} (multi-GPU)")

        # Extract tags from dataset
        item_indices = []
        original_tags = []
        current_tags = []

        for i, item in enumerate(data):
            tag = item.get("summary_words", [])[level_idx] if level_idx < len(item.get("summary_words", [])) else ""
            tag = tag.strip() if tag else ""
            if tag:
                item_indices.append(i)
                original_tags.append(tag)
                current_tags.append(tag)

        print(f"Items with level-{level_idx} tags: {len(current_tags)}")

        if not current_tags:
            return {}, [], np.zeros((0, 0), dtype=np.float32), {
                "level_idx": level_idx,
                "items_with_tags": 0,
                "iterations_run": 0,
                "final_changed_count": 0,
                "num_raw_unique_tags": 0,
                "num_final_unique_tags": 0,
            }

        unique_raw_tags = sorted(set(original_tags))
        print(f"Unique raw tags: {len(unique_raw_tags)}")

        if parent_canonical_tags and parent_embeddings is None:
            print("Encoding parent tags...")
            parent_embeddings = self.encode_batch(parent_canonical_tags, batch_size=batch_size)

        iterations_run = 0

        for iteration in range(max_iterations):
            iterations_run += 1
            unique_current_tags = sorted(set(current_tags))

            print(f"\nIteration {iteration + 1}/{max_iterations}")
            print(f"Unique current tags: {len(unique_current_tags)}")

            # Encode tags
            embeddings = self.encode_batch(unique_current_tags, batch_size=batch_size)

            # Cluster
            n_clusters = max(1, math.ceil(len(unique_current_tags) / cluster_size_target))
            print(f"Clustering into {n_clusters} clusters...")
            clusters, centroids = self.cluster_keywords_balanced(
                unique_current_tags,
                embeddings,
                n_clusters=n_clusters,
            )

            # Prepare cluster batches for each GPU
            cluster_list = sorted(clusters.items())
            clusters_per_gpu = math.ceil(len(cluster_list) / self.num_gpus)

            cluster_batches = [[] for _ in range(self.num_gpus)]
            for batch_idx, (cluster_id, cluster_terms) in enumerate(cluster_list):
                gpu_idx = batch_idx // clusters_per_gpu % self.num_gpus

                top_parent_tags = None
                if parent_canonical_tags and parent_embeddings is not None:
                    top_parent_tags = self.find_top_k_closest_tags(
                        centroids[cluster_id],
                        parent_embeddings,
                        parent_canonical_tags,
                        k=5,
                    )

                cluster_batches[gpu_idx].append((cluster_id, cluster_terms, top_parent_tags))

            # Dispatch to persistent workers
            active_ranks = []
            print(f"Dispatching {len(cluster_list)} clusters to {self.num_gpus} GPUs...")
            for rank, cluster_batch in enumerate(cluster_batches):
                if not cluster_batch:
                    continue
                self._input_queues[rank].put(cluster_batch)
                active_ranks.append(rank)

            # Collect results
            cluster_debug_rows = []
            mapping_list = []
            for _ in range(len(active_ranks)):
                rank, results = self._output_queue.get()

                # Check if this is an error message (string) instead of results (list)
                if isinstance(results, str):
                    print(f"ERROR from GPU {rank}: {results}")
                    raise RuntimeError(f"Worker process {rank} failed: {results}")

                print(f"Received {len(results)} cluster results from GPU {rank}")
                for cluster_mapping, cluster_debug in results:
                    mapping_list.append(cluster_mapping)
                    cluster_debug_rows.append(cluster_debug)

            # Merge and apply
            iter_mapping = self.merge_mappings(mapping_list, universe_terms=unique_current_tags)

            prev_tags = current_tags[:]
            current_tags = [iter_mapping.get(tag, tag) for tag in current_tags]

            # Update dataset
            for pos, item_idx in enumerate(item_indices):
                item = data[item_idx]
                if "summary_words" not in item:
                    item["summary_words"] = []
                while len(item["summary_words"]) <= level_idx:
                    item["summary_words"].append("")
                item["summary_words"][level_idx] = current_tags[pos]

            # Check convergence
            prev_unique = set(prev_tags)
            new_unique = set(current_tags)
            unique_tags_changed = len(prev_unique - new_unique)

            print(f"Unique tags changed: {unique_tags_changed}")

            if unique_tags_changed < threshold:
                print(f"Converged!")
                break

        # Final encoding
        final_canonical_tags = sorted(set(current_tags))
        print(f"Final unique tags: {len(final_canonical_tags)}")
        final_embeddings = self.encode_batch(final_canonical_tags, batch_size=batch_size)

        # Build mapping
        raw_to_final_mapping = {}
        votes: Dict[str, Counter] = defaultdict(Counter)
        for raw_tag, final_tag in zip(original_tags, current_tags):
            votes[raw_tag][final_tag] += 1

        for raw_tag, counter in votes.items():
            raw_to_final_mapping[raw_tag] = counter.most_common(1)[0][0]

        level_stats = {
            "level_idx": level_idx,
            "items_with_tags": len(current_tags),
            "iterations_run": iterations_run,
            "final_changed_count": 0,
            "num_raw_unique_tags": len(unique_raw_tags),
            "num_final_unique_tags": len(final_canonical_tags),
        }

        return raw_to_final_mapping, final_canonical_tags, final_embeddings, level_stats

    def refine_hierarchy(
        self,
        data_file: str,
        output_file: str,
        updated_data_file: Optional[str] = None,
        num_levels: Optional[int] = None,
        target_levels: Optional[List[int]] = None,
        threshold: int = 10,
        max_iterations: int = 5,
        batch_size: int = 32,
        cluster_size_target: int = 30,
    ) -> Dict:
        """Refine hierarchy across all levels with multi-GPU support."""
        print("Loading data...")
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        print(f"Loaded {len(data)} items")

        if num_levels is None:
            max_levels = 0
            for item in data:
                if isinstance(item.get("summary_words"), list):
                    max_levels = max(max_levels, len(item["summary_words"]))
            num_levels = max_levels

        levels_to_process = target_levels if target_levels is not None else list(range(num_levels))
        print(f"Processing levels: {levels_to_process}")

        mappings: List[Dict[str, str]] = []
        canonical_tags_per_level: List[List[str]] = []
        embeddings_per_level: List[np.ndarray] = []
        level_stats: List[Dict[str, int]] = []

        parent_tags = None
        parent_embeddings = None

        self.start_workers()

        for level_idx in levels_to_process:
            mapping, final_tags, final_embs, stats = self.stabilize_level_multiprocess(
                level_idx=level_idx,
                data=data,
                parent_canonical_tags=parent_tags,
                parent_embeddings=parent_embeddings,
                threshold=threshold,
                max_iterations=max_iterations,
                batch_size=batch_size,
                cluster_size_target=cluster_size_target,
            )

            mappings.append(mapping)
            canonical_tags_per_level.append(final_tags)
            embeddings_per_level.append(final_embs)
            level_stats.append(stats)

            parent_tags = final_tags
            parent_embeddings = final_embs

            # Save intermediate results after each level
            print(f"\n===== Saving results after level {level_idx} =====")
            
            interim_updated_data_file = updated_data_file if updated_data_file is not None else output_file.replace(".json", "") + "_updated.jsonl"
            
            # Save updated data
            print(f"Saving updated dataset to: {interim_updated_data_file}")
            with open(interim_updated_data_file, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            # Save taxonomy
            intermediate_results = {
                "mappings": mappings,
                "canonical_tags": canonical_tags_per_level,
                "num_tags_per_level": [len(tags) for tags in canonical_tags_per_level],
                "level_stats": level_stats,
                "current_level": level_idx,
                "input_file": data_file,
                "updated_data_file": interim_updated_data_file,
            }
            
            print(f"Saving taxonomy to: {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(intermediate_results, f, ensure_ascii=False, indent=2)

        self.stop_workers()

        # Save results
        if updated_data_file is None:
            updated_data_file = output_file.replace(".json", "") + "_updated.jsonl"

        print(f"\nSaving updated dataset to: {updated_data_file}")
        with open(updated_data_file, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        results = {
            "mappings": mappings,
            "canonical_tags": canonical_tags_per_level,
            "num_tags_per_level": [len(tags) for tags in canonical_tags_per_level],
            "level_stats": level_stats,
            "input_file": data_file,
            "updated_data_file": updated_data_file,
        }

        print(f"Saving taxonomy to: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Taxonomy Refinement (Multi-GPU)")

    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp",
        help="Dataset name (e.g., sports, toys, beauty, yelp)",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input JSONL file (default: ./data/{dataset}_tid_draft.jsonl)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output taxonomy JSON file (default: ./data/{dataset}_canonical_taxonomy.json)",
    )

    parser.add_argument(
        "--updated_data_file",
        type=str,
        default=None,
        help="Output JSONL file with refined summary_words (default: ./data/{dataset}_tid_updated.jsonl)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-9B",
        help="LLM model name",
    )

    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model name",
    )

    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=list(range(torch.cuda.device_count())),
        help="GPU IDs to use",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Convergence threshold",
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=15,
        help="Maximum iterations per level",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Embedding batch size",
    )

    parser.add_argument(
        "--cluster_size_target",
        type=int,
        default=32,
        help="Target cluster size",
    )

    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Specific hierarchy levels to process (e.g., 0 1 means process only levels 0 and 1)",
    )

    args = parser.parse_args()

    data_name = args.dataset
    input_file = args.input_file or f"./data/{data_name}_tid_draft.jsonl"
    output_file = args.output_file or f"./data/{data_name}_canonical_taxonomy.json"
    updated_data_file = args.updated_data_file or f"./data/{data_name}_tid_updated.jsonl"

    refiner = TaxonomyRefinerMultiGPU(
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        gpu_ids=args.gpu_ids,
        data_name=data_name,
    )

    refiner.refine_hierarchy(
        data_file=input_file,
        output_file=output_file,
        updated_data_file=updated_data_file,
        target_levels=args.levels,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        cluster_size_target=args.cluster_size_target,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()