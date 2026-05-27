import torch
import torch.nn.functional as F
import argparse
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    # Qwen's model card uses last-token pooling.
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

class QwenEmbeddingEncoder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        device: str | None = None,
        use_fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",   # recommended in the model card
        )

        dtype = torch.float16 if (use_fp16 and self.device.startswith("cuda")) else torch.float32

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_length: int = 8192,
        normalize: bool = True,
    ) -> torch.Tensor:
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), total=num_batches, desc="Encoding"):
            batch_texts = texts[i:i + batch_size]

            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])

            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hierarchical Taxonomy Refinement")
    parser.add_argument("--dataset", type=str, default="toys")
    args = parser.parse_args()  

    print(f"Loading items from {args.dataset}...")
    total_items = load_json(f"./data/{args.dataset}.item.json") 
    total_disc = ["Null point"] * int(len(total_items) + 1)

    print(f"Preparing descriptions for {len(total_items)} items...")

    if args.dataset == "yelp":
        for cur_id in total_items : 
            total_disc[int(cur_id)] = total_items[cur_id].get("title", "")

    else : 
        for cur_id in total_items : 
            total_disc[int(cur_id)] = "Title: " + total_items[cur_id].get("title", "") + "/ Description: " + total_items[cur_id].get("description", "")[:200]

    print("Initializing encoder...")
    encoder = QwenEmbeddingEncoder(device="cuda:0")
    print(f"Encoding {len(total_disc)} descriptions...")
    emb = encoder.encode(total_disc, batch_size=32)

    print(f"Saving embeddings to ./data/{args.dataset}_embeddings.pt...")
    torch.save(emb, f"./data/{args.dataset}_embeddings.pt")
    print("Done!")