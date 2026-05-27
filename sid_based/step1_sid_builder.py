"""
RQ-VAE with:
  1. Multi-resolution codebooks (decreasing sizes per level)
  2. Gradient-based codebook updates (no EMA)
  3. Co-occurrence contrastive loss (pull co-purchased items closer)

Usage:
    python codebook_generation_v2.py --dataset toys --epochs 100 --lr 1e-3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import json
from tqdm import tqdm


class RQVAEEncoder(nn.Module):
    """MLP encoder that projects item embeddings before quantization."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResidualQuantizer(nn.Module):
    """
    Learnable Residual Quantizer with multi-resolution codebooks.
    Codebooks are updated via gradient descent (optimizer).
    
    Anti-collapse mechanism:
      Dead code restart (reinitialize unused codes from data)
    
    Args:
        codebook_sizes: list of codebook sizes per level, e.g. [1024, 512, 256]
        dim: embedding dimension
        dead_code_threshold: reset codes unused for this many steps
    """

    def __init__(self, codebook_sizes, dim, dead_code_threshold=50):
        super().__init__()
        self.codebook_sizes = codebook_sizes
        self.n_levels = len(codebook_sizes)
        self.dim = dim
        self.dead_code_threshold = dead_code_threshold

        # Codebooks — updated via optimizer (gradient-based)
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(k, dim) * 0.02)
            for k in codebook_sizes
        ])

        # Dead code tracking buffers
        for level, k in enumerate(codebook_sizes):
            self.register_buffer(f"steps_since_used_{level}", torch.zeros(k, dtype=torch.long))

    @torch.no_grad()
    def init_codebooks_from_data(self, z):
        """Initialize codebooks using k-means on encoder output (residual for each level)."""
        residual = z
        for level in range(self.n_levels):
            K = self.codebook_sizes[level]
            N = residual.shape[0]
            # K-means++ initialization
            indices = [torch.randint(0, N, (1,)).item()]
            for _ in range(1, K):
                # Compute distances to nearest existing centroid
                centroids = residual[indices]  # [len(indices), D]
                dists = torch.cdist(residual, centroids)  # [N, len(indices)]
                min_dists = dists.min(dim=1).values  # [N]
                # Sample proportional to distance squared
                probs = min_dists ** 2
                probs /= probs.sum()
                idx = torch.multinomial(probs, 1).item()
                indices.append(idx)
            self.codebooks[level].data.copy_(residual[indices])
            # Assign and compute residual for next level
            cb_dists = torch.cdist(residual, self.codebooks[level].data)
            assignments = cb_dists.argmin(dim=1)
            residual = residual - self.codebooks[level].data[assignments]
            print(f"  Level {level}: initialized {K} codebook entries via k-means++")

    def _dead_code_restart(self, level, indices, residual):
        """Restart dead codes by reinitializing them from random data points."""
        steps_since = getattr(self, f"steps_since_used_{level}")
        codebook = self.codebooks[level]
        K = codebook.shape[0]
        B = indices.shape[0]

        # Track usage
        counts = torch.bincount(indices, minlength=K)
        used_mask = counts > 0
        steps_since[used_mask] = 0
        steps_since[~used_mask] += 1

        # Restart dead codes
        dead_mask = steps_since >= self.dead_code_threshold
        n_dead = dead_mask.sum().item()
        if n_dead > 0 and B > 0:
            replace_idx = torch.randint(0, B, (n_dead,), device=residual.device)
            codebook.data[dead_mask] = residual[replace_idx].detach()
            steps_since[dead_mask] = 0

        return n_dead

    def forward(self, z):
        """
        Quantize z using residual quantization (all levels).
        
        Returns:
            z_q: quantized representation
            codes: [B, n_levels] codebook indices
            commitment_loss: commitment loss (encoder gradients)
            codebook_loss: codebook loss (codebook gradients)
            usage_stats: dict with codebook utilization info
        """
        residual = z
        z_q = torch.zeros_like(z)
        codes = []
        commitment_loss = 0.0
        codebook_loss = 0.0
        total_dead = 0

        for level in range(self.n_levels):
            codebook = self.codebooks[level]  # [K, D]
            K = codebook.shape[0]

            # Compute distances
            dists = torch.cdist(residual.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)  # [B, K]
            indices = dists.argmin(dim=1)  # [B]
            codes.append(indices)

            # Quantized vectors
            quantized = codebook[indices]  # [B, D]

            # PLUM-style RQ loss: β||r_l - sg[e*_l]||² + ||sg[r_l] - e*_l||²
            commitment_loss += F.mse_loss(residual, quantized.detach())  # → encoder
            codebook_loss += F.mse_loss(residual.detach(), quantized)     # → codebook

            # Dead code restart
            if self.training:
                n_dead = self._dead_code_restart(level, indices, residual)
                total_dead += n_dead

            # Straight-through estimator
            quantized_st = residual + (quantized - residual).detach()
            z_q = z_q + quantized_st
            residual = residual - quantized_st

        usage_stats = {"dead_codes_restarted": total_dead}

        return z_q, torch.stack(codes, dim=1), commitment_loss, codebook_loss, usage_stats

    @torch.no_grad()
    def encode(self, z):
        """Encode without gradients (inference)."""
        residual = z
        codes = []
        for level in range(self.n_levels):
            codebook = self.codebooks[level]
            dists = torch.cdist(residual.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
            indices = dists.argmin(dim=1)
            codes.append(indices)
            residual = residual - codebook[indices]
        return torch.stack(codes, dim=1)

    @torch.no_grad()
    def decode(self, codes):
        """Decode codes back to embeddings."""
        B = codes.shape[0]
        z_q = torch.zeros(B, self.dim, device=codes.device)
        for level in range(self.n_levels):
            valid = codes[:, level] >= 0
            if valid.any():
                z_q[valid] += self.codebooks[level][codes[valid, level]]
        return z_q


class RQVAE(nn.Module):
    """
    Full RQ-VAE model with encoder, quantizer, and decoder.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, codebook_sizes,
                 dead_code_threshold=50):
        super().__init__()
        self.encoder = RQVAEEncoder(input_dim, hidden_dim, latent_dim)
        self.quantizer = ResidualQuantizer(
            codebook_sizes, latent_dim,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, codes, commitment_loss, codebook_loss, usage_stats = self.quantizer(z)
        x_recon = self.decoder(z_q)
        return x_recon, z, z_q, codes, commitment_loss, codebook_loss, usage_stats

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        return self.quantizer.encode(z)


def build_cooccurrence_pairs(dataset_name):
    """
    Load co-purchase neighbors and build co-occurrence pairs.
    Returns: dict mapping item_idx (1-indexed) -> list of co-occurring item_idxs
    """
    copurchase_file = f"../data/{dataset_name}_collaborative_neighbors.json"
    try:
        with open(copurchase_file, 'r') as f:
            copurchase = json.load(f)
        # Keep 1-indexed (embeddings[0] is a placeholder)
        pairs = {}
        for item_id, neighbors in copurchase.items():
            idx = int(item_id)
            neighbor_idxs = [int(n) for n in neighbors[:5] if int(n) >= 1]
            if neighbor_idxs:
                pairs[idx] = neighbor_idxs
        return pairs
    except FileNotFoundError:
        print(f"Warning: {copurchase_file} not found, skipping contrastive loss")
        return {}


def train_rqvae_with_dedicated_contrastive(
    embeddings: torch.Tensor,
    codebook_sizes: list,
    cooccurrence_pairs: dict,
    hidden_dim: int = 512,
    latent_dim: int = 256,
    batch_size: int = 1024,
    epochs: int = 100,
    lr: float = 1e-3,
    beta_commitment: float = 0.25,
    beta_contrastive: float = 0.1,
    device: str = "cuda",
    dead_code_threshold: int = 50,
):
    """
    Train RQ-VAE with dedicated contrastive batches built from co-occurrence pairs.
    Codebooks are updated via gradient descent (no EMA). All levels always active (no progressive masking).
    Codebooks are initialized via k-means++ on encoder output.
    """
    N, D = embeddings.shape
    model = RQVAE(D, hidden_dim, latent_dim, codebook_sizes,
                  dead_code_threshold=dead_code_threshold).to(device)

    # Initialize codebooks from data using k-means++
    print("Initializing codebooks via k-means++...")
    with torch.no_grad():
        init_z = model.encoder(embeddings[:min(N, 10000)].to(device))
        model.quantizer.init_codebooks_from_data(init_z)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-build pair list
    pair_list = []
    for idx, neighbors in cooccurrence_pairs.items():
        for n in neighbors:
            if n < N:
                pair_list.append((idx, n))
    pair_list = np.array(pair_list)
    print(f"Total co-occurrence pairs: {len(pair_list)}")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        total_contra = 0.0
        total_dead = 0
        n_batches = 0

        for i in range(0, N, batch_size):
            batch_idx = perm[i:i+batch_size]
            x = embeddings[batch_idx].to(device)

            x_recon, z, z_q, codes, commitment_loss, codebook_loss, usage_stats = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            # PLUM: L = L_recon + β * L_commit + L_codebook
            loss = recon_loss + beta_commitment * commitment_loss + codebook_loss

            # Contrastive: sample dedicated pairs
            contra_loss = torch.tensor(0.0, device=device)
            if len(pair_list) > 0:
                n_pairs = min(batch_size // 2, len(pair_list))
                pair_idx = np.random.choice(len(pair_list), n_pairs, replace=False)
                sampled_pairs = pair_list[pair_idx]

                anchor_emb = embeddings[sampled_pairs[:, 0]].to(device)
                pos_emb = embeddings[sampled_pairs[:, 1]].to(device)

                # Encode both
                z_anchor = model.encoder(anchor_emb)
                z_pos = model.encoder(pos_emb)

                # InfoNCE: positives are at diagonal
                z_a_norm = F.normalize(z_anchor, dim=1)
                z_p_norm = F.normalize(z_pos, dim=1)
                sim = z_a_norm @ z_p_norm.T / 0.5  # [n_pairs, n_pairs]
                labels = torch.arange(n_pairs, device=device)
                contra_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
                loss = loss + beta_contrastive * contra_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contra += contra_loss.item()
            total_dead += usage_stats.get("dead_codes_restarted", 0)
            n_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            # Compute codebook utilization
            with torch.no_grad():
                codes = model.encode(embeddings[:min(5000, N)].to(device))
                utilization = []
                for level in range(len(codebook_sizes)):
                    used = codes[:, level].unique().numel()
                    utilization.append(f"L{level}:{used}/{codebook_sizes[level]}")
                z_gt = model.encoder(embeddings[:1000].to(device))
                recon = model.quantizer.decode(model.encode(embeddings[:1000].to(device)))
                quant_mse = F.mse_loss(recon, z_gt).item()
            print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | contra={total_contra/n_batches:.4f} | quant_mse={quant_mse:.6f} | "
                  f"dead_restarted={total_dead} | util=[{', '.join(utilization)}]")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toys")
    parser.add_argument("--codebook_sizes", type=str, default="1024,512,256",
                        help="Comma-separated codebook sizes (multi-resolution)")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3) # 1e-3
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--beta_commitment", type=float, default=0.25)
    parser.add_argument("--beta_contrastive", type=float, default=0.1) # 0.1
    parser.add_argument("--dead_code_threshold", type=int, default=30, help="Reset codes unused for this many steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    codebook_sizes = [int(x) for x in args.codebook_sizes.split(",")]
    print(f"Codebook sizes (multi-resolution): {codebook_sizes}")

    # Load embeddings
    embeddings = torch.load(f"../data/{args.dataset}_embeddings.pt", map_location="cpu").float()
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load co-occurrence pairs
    cooccurrence_pairs = build_cooccurrence_pairs(args.dataset)
    print(f"Loaded {len(cooccurrence_pairs)} items with co-occurrence info")

    # Train
    model = train_rqvae_with_dedicated_contrastive(
        embeddings=embeddings,
        codebook_sizes=codebook_sizes,
        cooccurrence_pairs=cooccurrence_pairs,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta_commitment=args.beta_commitment,
        beta_contrastive=args.beta_contrastive,
        device=args.device,
        dead_code_threshold=args.dead_code_threshold,
    )

    # Encode all items
    model.eval()
    with torch.no_grad():
        all_codes = []
        for i in range(0, len(embeddings), 2048):
            batch = embeddings[i:i+2048].to(args.device)
            codes = model.encode(batch)
            all_codes.append(codes.cpu())
        all_codes = torch.cat(all_codes, dim=0)

    print(f"\nFinal codes shape: {all_codes.shape}")
    print(f"First 5 codes:\n{all_codes[:5]}")

    # Check uniqueness
    code_tuples = [tuple(c.tolist()) for c in all_codes]
    n_unique = len(set(code_tuples))
    print(f"Unique SIDs: {n_unique}/{len(all_codes)} ({n_unique/len(all_codes)*100:.1f}%)")

    # Save
    output_path = f"./data/{args.dataset}_codebook.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(all_codes, f)
    print(f"Saved codes to {output_path}")

    # We don't need this at this time.
    # # Also save the model for later use
    # model_path = f"../data/{args.dataset}_rqvae_model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Saved model to {model_path}")