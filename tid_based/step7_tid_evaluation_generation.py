from collections import defaultdict
import re
import os
import json
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"


def load_test_data(file_path):
    """Load test data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    return sft_data


def prepare_batch_prompts(batch_data, num_keywords):
    """Prepare prompts for batch data — aligned with step4_prediction_dataset_new.py"""
    batch_prompts = []
    batch_metadata = []

    for d in batch_data:
        # Build context: input item + all output items + valid ground truth item
        l = d["input"] + d["output"]
        l += "Item keywords: [" + ", ".join(d["valid_ground_truth_tid"]) + "]"
        if "title" in d["valid_ground_truth_msg"]:
            temp = d["valid_ground_truth_msg"]["title"]
            l += f" Title: {temp}.\n"
        else:
            l += f" Title: None.\n"

        # Instruction aligned with training data
        instruction = (
            f"Given a user's historical item interaction sequence, predict the keywords of the next item "
            f"the user is most likely to interact with. Each item in the sequence is represented by exactly "
            f"{num_keywords} keywords enclosed in square brackets []. The items are listed in chronological order.\n"
        )

        prompt = instruction + l + "Item keywords: ["

        messages = [{"role": "user", "content": prompt}]
        batch_prompts.append(messages)
        batch_metadata.append({
            'original_data': d,
            'prompt': prompt
        })

    return batch_prompts, batch_metadata


def process_single_gpu(rank, data_slice, output_queue, model_name, num_keywords, batch_size=8):
    """Process data slice on each GPU and only save generated outputs"""
    print(f"Rank {rank}: Initializing model and tokenizer...")

    torch.cuda.set_device(rank)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        dtype=torch.float16,
        device_map=f"cuda:{rank}"
    )
    model.eval()

    print(f"Rank {rank}: Model loaded, starting to process {len(data_slice)} test samples, batch_size={batch_size}")

    local_results = []

    for batch_start in tqdm(range(0, len(data_slice), batch_size), desc=f"GPU {rank}"):
        batch_end = min(batch_start + batch_size, len(data_slice))
        batch_data = data_slice[batch_start:batch_end]

        batch_prompts, batch_metadata = prepare_batch_prompts(batch_data, num_keywords)

        batch_texts = []
        for messages in batch_prompts:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            batch_texts.append(text)

        model_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32768,
            return_attention_mask=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=30,
                do_sample=False,
                num_beams=20,
                num_return_sequences=20,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=False,
            )

        batch_results = process_batch_results(
            generated_ids, model_inputs, batch_metadata, tokenizer
        )
        local_results.extend(batch_results)

    output_queue.put((rank, local_results))
    print(f"Rank {rank}: Processing completed, processed {len(local_results)} samples in total")


def process_batch_results(generated_ids, model_inputs, batch_metadata, tokenizer):
    """Process batch generation results and keep raw generated outputs only"""
    batch_results = []

    num_sequences_per_sample = generated_ids.shape[0] // len(batch_metadata)

    for batch_idx, metadata in enumerate(batch_metadata):
        dic = metadata['original_data'].copy()

        raw_contents = []

        start_idx = batch_idx * num_sequences_per_sample
        end_idx = (batch_idx + 1) * num_sequences_per_sample

        for seq_idx in range(start_idx, end_idx):
            input_len = model_inputs.input_ids[batch_idx].shape[0]
            output_ids = generated_ids[seq_idx][input_len:].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            raw_contents.append(content)

        dic["prompt"] = metadata["prompt"]
        dic["raw_contents"] = raw_contents
        dic["num_generations"] = len(raw_contents)

        batch_results.append(dic)

    return batch_results


def detect_num_keywords_from_data(sft_data):
    """Detect the number of keywords from the first sample"""
    for d in sft_data:
        if "valid_ground_truth_tid" in d and len(d["valid_ground_truth_tid"]) > 0:
            return len(d["valid_ground_truth_tid"])
    return 3


def main(model_path, dataset_name):
    ckpts = [model_path]

    for ckpt in ckpts:
        model_name = f"{ckpt}"

        test_file = f"./train_data/{dataset_name}_tid_next_item_prediction_sft.json"

        print(f"Loading test data: {test_file}")
        sft_data = load_test_data(test_file)

        sft_data = sft_data[:5000]

        print(f"Loaded {len(sft_data)} test samples")

        print(f"Generating outputs for {len(sft_data)} samples")

        num_keywords = detect_num_keywords_from_data(sft_data)
        print(f"Detected {num_keywords} keywords per item")

        num_gpus = 2
        print(f"Using {num_gpus} GPUs")

        chunk_size = len(sft_data) // num_gpus
        data_chunks = []
        for i in range(num_gpus):
            start_idx = i * chunk_size
            end_idx = len(sft_data) if i == num_gpus - 1 else start_idx + chunk_size
            data_chunks.append(sft_data[start_idx:end_idx])

        print(f"Data split into {num_gpus} chunks, chunk sizes: {[len(chunk) for chunk in data_chunks]}")

        processes = []
        output_queue = mp.Queue()

        print("Starting multi-process generation...")
        start_time = time.time()

        batch_size = 4

        for rank in range(num_gpus):
            p = mp.Process(
                target=process_single_gpu,
                args=(rank, data_chunks[rank], output_queue, model_name, num_keywords, batch_size)
            )
            processes.append(p)
            p.start()

        all_results = []

        for _ in range(num_gpus):
            rank, local_results = output_queue.get()
            print(f"Received {len(local_results)} results from GPU {rank}")
            all_results.extend(local_results)

        for p in processes:
            p.join()

        end_time = time.time()
        print(f"Multi-GPU generation completed, total time: {end_time - start_time:.2f} seconds")

        os.makedirs("./results", exist_ok=True)
        output_file = f"./results/{dataset_name}_generation_outputs_tid.json"
        print(f"\nSaving generation outputs to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f :
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\nGeneration completed! Total samples processed: {len(all_results)}")


if __name__ == "__main__":
    import argparse

    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Generation script for recommendation model')
    parser.add_argument('--dataset_name', type=str, default='toys',
                        help='Name of the dataset to evaluate on')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the model checkpoint (default: model_{dataset_name}_tid_sft)')
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"model_{args.dataset_name}_tid_sft"

    print(args.model_path, args.dataset_name)

    main(args.model_path, args.dataset_name)
