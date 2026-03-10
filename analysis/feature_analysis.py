"""
Sparse autoencoder feature analysis for GPT-2 residual stream.

Trains a sparse autoencoder on layer-6 residual stream activations from the FP model,
then applies it to quantized activations to measure feature survival.

Architecture: Linear(d_model, 4*d_model) + ReLU + Linear(4*d_model, d_model)
Loss: MSE(reconstruction) + lambda * L1(hidden)
"""

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_models, tokenize_prompts, ALL_PROMPTS
from utils.metrics import feature_survival_rate


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for finding monosemantic features in residual stream.

    Architecture:
        encoder: d_model -> d_hidden (with ReLU)
        decoder: d_hidden -> d_model
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.decoder = nn.Linear(d_hidden, d_model)

        # Xavier init for better training
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input activations, shape (..., d_model).

        Returns:
            (reconstruction, hidden): Reconstructed input and hidden activations.
        """
        hidden = F.relu(self.encoder(x))
        reconstruction = self.decoder(hidden)
        return reconstruction, hidden


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_residual_stream(
    model,
    tokens: torch.Tensor,
    layer: int = 6,
    batch_size: int = 8,
) -> torch.Tensor:
    """
    Extract residual stream activations at a specific layer.

    Returns:
        Tensor of shape (n_prompts * seq_len, d_model).
    """
    device = next(model.parameters()).device
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_acts = []

    for start in tqdm(range(0, len(tokens), batch_size), desc=f"Layer {layer} residuals"):
        batch = tokens[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=[hook_name])
        acts = cache[hook_name].cpu()  # (batch, seq_len, d_model)
        all_acts.append(acts.reshape(-1, acts.shape[-1]))

    return torch.cat(all_acts, dim=0)


# ---------------------------------------------------------------------------
# SAE Training
# ---------------------------------------------------------------------------

def train_sparse_autoencoder(
    activations: torch.Tensor,
    d_model: int,
    d_hidden: int = 3072,
    l1_lambda: float = 1e-3,
    lr: float = 1e-3,
    n_epochs: int = 1000,
    batch_size: int = 2048,
    device: str = "cuda",
) -> SparseAutoencoder:
    """
    Train a sparse autoencoder on residual stream activations.

    Args:
        activations: Shape (n_tokens, d_model).
        d_model: Model hidden dimension.
        d_hidden: SAE hidden dimension (4 * d_model).
        l1_lambda: L1 sparsity penalty weight.
        lr: Learning rate.
        n_epochs: Number of training epochs.
        batch_size: Mini-batch size.
        device: Training device.

    Returns:
        Trained SparseAutoencoder.
    """
    sae = SparseAutoencoder(d_model, d_hidden).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    activations = activations.to(device)
    n_samples = activations.shape[0]

    print(f"\nTraining SAE: {d_model} -> {d_hidden} -> {d_model}")
    print(f"  Samples: {n_samples}, Epochs: {n_epochs}, L1 lambda: {l1_lambda}")

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=device)
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            x = activations[idx]

            recon, hidden = sae(x)
            mse_loss = F.mse_loss(recon, x)
            l1_loss = hidden.abs().mean()
            loss = mse_loss + l1_lambda * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            n_batches += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / n_batches
            avg_mse = total_mse / n_batches
            avg_l1 = total_l1 / n_batches
            # Sparsity: fraction of dead neurons
            with torch.no_grad():
                sample = activations[:min(1024, n_samples)]
                _, h = sae(sample)
                alive = (h > 0).float().mean(dim=0)  # fraction of samples activating each neuron
                dead_frac = (alive == 0).float().mean().item()
            print(f"  Epoch {epoch+1:4d}: loss={avg_loss:.6f} mse={avg_mse:.6f} l1={avg_l1:.6f} dead={dead_frac:.2%}")

    return sae


# ---------------------------------------------------------------------------
# Feature analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_features(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    tokens_flat: torch.Tensor,
    tokenizer,
    top_k_features: int = 20,
    top_k_tokens: int = 10,
    device: str = "cuda",
) -> list[dict]:
    """
    Identify top features and the tokens that activate them most.

    Args:
        sae: Trained sparse autoencoder.
        activations: Residual stream activations, shape (n_tokens, d_model).
        tokens_flat: Flattened token IDs, shape (n_tokens,).
        tokenizer: Tokenizer for decoding.
        top_k_features: Number of top features to analyze.
        top_k_tokens: Number of top-activating tokens per feature.

    Returns:
        List of feature dicts with keys: feature_idx, mean_activation, top_tokens, top_token_ids.
    """
    sae.eval()
    activations = activations.to(device)

    # Get hidden activations for all tokens
    _, hidden = sae(activations)  # (n_tokens, d_hidden)
    hidden = hidden.cpu()

    # Mean activation per feature
    mean_acts = hidden.mean(dim=0)  # (d_hidden,)
    top_features = mean_acts.topk(top_k_features).indices.tolist()

    features = []
    for feat_idx in top_features:
        feat_acts = hidden[:, feat_idx]  # (n_tokens,)
        top_indices = feat_acts.topk(top_k_tokens).indices.tolist()
        top_token_ids = tokens_flat[top_indices].tolist()

        # Decode tokens
        top_tokens_decoded = [tokenizer.decode([tid]) for tid in top_token_ids]

        features.append({
            "feature_idx": feat_idx,
            "mean_activation": mean_acts[feat_idx].item(),
            "max_activation": feat_acts.max().item(),
            "top_token_ids": top_token_ids,
            "top_tokens": top_tokens_decoded,
            "top_activations": feat_acts[top_indices].tolist(),
        })

    return features


@torch.no_grad()
def compare_feature_survival(
    sae: SparseAutoencoder,
    fp_acts: torch.Tensor,
    q_acts: torch.Tensor,
    fp_tokens_flat: torch.Tensor,
    q_tokens_flat: torch.Tensor,
    top_k_features: int = 20,
    top_k_tokens: int = 10,
    device: str = "cuda",
) -> tuple[float, list[float], list[dict]]:
    """
    Compare which features survive quantization by checking if the same
    tokens activate them in both FP and quantized models.

    Returns:
        (survival_rate, per_feature_overlap, comparison_details)
    """
    sae.eval()

    # Get hidden activations from both models
    _, fp_hidden = sae(fp_acts.to(device))
    _, q_hidden = sae(q_acts.to(device))
    fp_hidden = fp_hidden.cpu()
    q_hidden = q_hidden.cpu()

    # Top features by FP mean activation
    mean_acts = fp_hidden.mean(dim=0)
    top_features = mean_acts.topk(top_k_features).indices.tolist()

    fp_top_tokens_list = []
    q_top_tokens_list = []
    comparison = []

    for feat_idx in top_features:
        fp_feat = fp_hidden[:, feat_idx]
        q_feat = q_hidden[:, feat_idx]

        fp_top_idx = fp_feat.topk(top_k_tokens).indices.tolist()
        q_top_idx = q_feat.topk(top_k_tokens).indices.tolist()

        fp_top_tids = fp_tokens_flat[fp_top_idx].tolist()
        q_top_tids = q_tokens_flat[q_top_idx].tolist()

        fp_top_tokens_list.append(fp_top_tids)
        q_top_tokens_list.append(q_top_tids)

        overlap = len(set(fp_top_tids) & set(q_top_tids)) / len(set(fp_top_tids))

        comparison.append({
            "feature_idx": feat_idx,
            "overlap": overlap,
            "fp_activation_mean": fp_feat.mean().item(),
            "q_activation_mean": q_feat.mean().item(),
            "correlation": torch.corrcoef(torch.stack([fp_feat, q_feat]))[0, 1].item(),
        })

    survival, per_feature = feature_survival_rate(fp_top_tokens_list, q_top_tokens_list)
    return survival, per_feature, comparison


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feature_analysis(
    fp_features: list[dict],
    survival: float,
    per_feature_overlap: list[float],
    comparison: list[dict],
    tokenizer,
    save_dir: str = "results",
):
    """Generate feature analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Feature activation heatmap — top tokens for top features
    n_features = len(fp_features)
    n_tokens = min(10, len(fp_features[0]["top_tokens"]))

    fig, ax = plt.subplots(figsize=(14, 8))
    data = np.zeros((n_features, n_tokens))
    xlabels = []
    ylabels = []

    for i, feat in enumerate(fp_features):
        ylabels.append(f"F{feat['feature_idx']}")
        for j in range(n_tokens):
            data[i, j] = feat["top_activations"][j]
            if i == 0:
                xlabels.append(repr(feat["top_tokens"][j]).strip("'"))

    # Only use first feature's tokens as x-labels (they vary per feature)
    sns.heatmap(data, annot=False, cmap="viridis", ax=ax)
    ax.set_xlabel("Token rank (by activation strength)")
    ax.set_ylabel("Feature")
    ax.set_yticklabels(ylabels, rotation=0)
    ax.set_title("Top-20 SAE Features: Activation Strengths on Top Tokens (FP model)")
    fig.tight_layout()
    path = os.path.join(save_dir, "feature_activation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 2. Feature top tokens display
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis("off")
    cell_text = []
    for feat in fp_features:
        tokens_str = ", ".join([repr(t).strip("'") for t in feat["top_tokens"][:5]])
        cell_text.append([
            f"F{feat['feature_idx']}",
            f"{feat['mean_activation']:.4f}",
            tokens_str,
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Feature", "Mean Act.", "Top-5 Tokens"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title(f"SAE Feature Inventory (FP model)", fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(save_dir, "feature_inventory.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 3. Feature survival
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-feature overlap bar chart
    feature_labels = [f"F{c['feature_idx']}" for c in comparison]
    overlaps = per_feature_overlap
    colors = ["#2ecc71" if o >= 0.8 else "#e74c3c" for o in overlaps]
    x_pos = range(len(feature_labels))
    axes[0].bar(x_pos, overlaps, color=colors)
    axes[0].axhline(y=0.8, color="black", linestyle="--", alpha=0.5, label="80% threshold")
    axes[0].set_ylabel("Token Overlap (FP vs 4-bit)")
    axes[0].set_title(f"Feature Survival: {survival:.0%} survive (overlap >= 80%)")
    axes[0].set_xticks(list(x_pos))
    axes[0].set_xticklabels(feature_labels, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Correlation scatter
    corrs = [c["correlation"] for c in comparison]
    axes[1].bar(x_pos, corrs, color="#3498db")
    axes[1].set_ylabel("Activation Correlation (FP vs 4-bit)")
    axes[1].set_title("Per-Feature Activation Correlation")
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(feature_labels, rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Monosemantic Feature Survival under 4-bit Quantization", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "feature_survival.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_feature_analysis(
    fp_model=None,
    q_model=None,
    tokens=None,
    save_dir: str = "results",
    layer: int = 6,
    n_epochs: int = 1000,
    batch_size: int = 8,
    sae=None,
) -> dict:
    """Full feature analysis pipeline."""
    if fp_model is None or q_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp_model, q_model = load_models(device=device)
    else:
        device = str(next(fp_model.parameters()).device)

    if tokens is None:
        tokens = tokenize_prompts(fp_model)

    d_model = fp_model.cfg.d_model
    d_hidden = 4 * d_model if d_model <= 1024 else 2 * d_model

    # Extract residual stream activations
    print(f"\n=== Extracting layer {layer} residual stream (FP) ===")
    fp_acts = extract_residual_stream(fp_model, tokens, layer=layer, batch_size=batch_size)

    print(f"\n=== Extracting layer {layer} residual stream (4-bit) ===")
    q_acts = extract_residual_stream(q_model, tokens, layer=layer, batch_size=batch_size)

    # Flatten tokens for mapping activations back to token IDs
    tokens_flat = tokens.reshape(-1)
    # Both models use same tokens so same flat array
    assert fp_acts.shape[0] == tokens_flat.shape[0], \
        f"Activation count {fp_acts.shape[0]} != token count {tokens_flat.shape[0]}"

    # Train SAE on FP activations (or reuse provided one)
    if sae is None:
        print("\n=== Training Sparse Autoencoder ===")
        sae = train_sparse_autoencoder(
            fp_acts, d_model, d_hidden=d_hidden,
            n_epochs=n_epochs, device=device,
        )
    else:
        print("\n=== Reusing pre-trained SAE ===")

    # Analyze FP features
    print("\n=== Analyzing FP features ===")
    tokenizer = fp_model.tokenizer
    fp_features = analyze_features(
        sae, fp_acts, tokens_flat, tokenizer,
        top_k_features=20, top_k_tokens=10, device=device,
    )

    # Compare survival
    print("\n=== Comparing feature survival (FP vs 4-bit) ===")
    survival, per_feature_overlap, comparison = compare_feature_survival(
        sae, fp_acts, q_acts, tokens_flat, tokens_flat,
        top_k_features=20, top_k_tokens=10, device=device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"SAE trained on layer {layer} residual stream")
    print(f"d_model={d_model}, d_hidden={d_hidden}")
    print(f"\nFeature survival rate: {survival:.1%}")
    print(f"Mean feature overlap: {np.mean(per_feature_overlap):.3f}")
    print(f"Mean activation correlation: {np.mean([c['correlation'] for c in comparison]):.3f}")

    print(f"\nTop-5 features (FP):")
    for feat in fp_features[:5]:
        tokens_str = ", ".join([repr(t) for t in feat["top_tokens"][:5]])
        print(f"  F{feat['feature_idx']}: mean_act={feat['mean_activation']:.4f}, tokens=[{tokens_str}]")

    plot_feature_analysis(fp_features, survival, per_feature_overlap, comparison, tokenizer, save_dir=save_dir)

    del fp_acts, q_acts

    return {
        "sae": sae,
        "fp_features": fp_features,
        "survival_rate": survival,
        "per_feature_overlap": per_feature_overlap,
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_feature_analysis()
