"""
Attention pattern comparison between FP and 4-bit quantized GPT-2.

For each of the 144 attention heads (12 layers x 12 heads):
  - Extract attention patterns on 50 prompts
  - Compute Jensen-Shannon divergence between FP and quantized patterns
  - Identify most/least affected heads
  - Generate 12x12 heatmap of divergence
  - Check if induction heads (layers 5-6) and previous-token heads survive
"""

import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_models, tokenize_prompts, ALL_PROMPTS
from utils.metrics import jensen_shannon_divergence


# ---------------------------------------------------------------------------
# Attention pattern extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_patterns(
    model,
    tokens: torch.Tensor,
    n_prompts: int = 50,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Extract attention patterns for all heads on a subset of prompts.

    Args:
        model: HookedTransformer.
        tokens: All tokenized prompts (n_total, seq_len).
        n_prompts: Number of prompts to use (first n_prompts).
        batch_size: Batch size for inference.

    Returns:
        Dict mapping hook_name -> tensor of shape (n_prompts, n_heads, seq_len, seq_len).
    """
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]

    subset = tokens[:n_prompts]
    all_patterns = {name: [] for name in hook_names}

    for start in tqdm(range(0, len(subset), batch_size), desc="Extracting attention"):
        batch = subset[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)

        for name in hook_names:
            all_patterns[name].append(cache[name].cpu())

    return {name: torch.cat(pats, dim=0) for name, pats in all_patterns.items()}


# ---------------------------------------------------------------------------
# JSD computation per head
# ---------------------------------------------------------------------------

def compute_head_jsd(
    fp_patterns: dict[str, torch.Tensor],
    q_patterns: dict[str, torch.Tensor],
    n_layers: int = 12,
    n_heads: int = 12,
) -> np.ndarray:
    """
    Compute mean Jensen-Shannon divergence per head between FP and quantized patterns.

    For each head, the attention pattern is a distribution over keys for each query.
    We compute JSD for each (prompt, query_position) pair, then average.

    Args:
        fp_patterns: FP attention patterns per layer.
        q_patterns: Quantized attention patterns per layer.

    Returns:
        jsd_matrix: Shape (n_layers, n_heads), mean JSD per head.
    """
    jsd_matrix = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        hook = f"blocks.{layer_idx}.attn.hook_pattern"
        fp = fp_patterns[hook]  # (n_prompts, n_heads, seq_len, seq_len)
        q = q_patterns[hook]

        for head_idx in range(n_heads):
            fp_head = fp[:, head_idx]  # (n_prompts, seq_len, seq_len)
            q_head = q[:, head_idx]

            # Flatten prompt and query dims, JSD over key dim
            fp_flat = fp_head.reshape(-1, fp_head.shape[-1])
            q_flat = q_head.reshape(-1, q_head.shape[-1])

            jsd = jensen_shannon_divergence(fp_flat, q_flat).mean().item()
            jsd_matrix[layer_idx, head_idx] = jsd

    return jsd_matrix


# ---------------------------------------------------------------------------
# Previous-token head detection
# ---------------------------------------------------------------------------

def compute_prev_token_score(patterns: dict[str, torch.Tensor], n_layers: int = 12, n_heads: int = 12) -> np.ndarray:
    """
    Score each head on how much attention it places on the previous token.

    A "previous token head" concentrates attention on position (i-1) for query position i.

    Returns:
        scores: Shape (n_layers, n_heads). Higher = more previous-token-like.
    """
    scores = np.zeros((n_layers, n_heads))

    for layer_idx in range(n_layers):
        hook = f"blocks.{layer_idx}.attn.hook_pattern"
        pat = patterns[hook]  # (n_prompts, n_heads, seq_len, seq_len)
        seq_len = pat.shape[-1]

        for head_idx in range(n_heads):
            head_pat = pat[:, head_idx]  # (n_prompts, seq_len, seq_len)
            # For each query position i >= 1, check attention on position i-1
            prev_attn = 0.0
            count = 0
            for i in range(1, seq_len):
                prev_attn += head_pat[:, i, i - 1].mean().item()
                count += 1
            scores[layer_idx, head_idx] = prev_attn / max(count, 1)

    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_attention_analysis(
    jsd_matrix: np.ndarray,
    prev_scores_fp: np.ndarray,
    prev_scores_q: np.ndarray,
    save_dir: str = "results",
):
    """Generate attention analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.1)

    # 1. JSD heatmap
    n_layers, n_heads = jsd_matrix.shape
    show_annot = n_layers * n_heads <= 200
    fig_w = max(10, n_heads * 0.8)
    fig_h = max(8, n_layers * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        jsd_matrix,
        annot=show_annot,
        fmt=".4f" if show_annot else "",
        cmap="YlOrRd",
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        ax=ax,
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Jensen-Shannon Divergence: FP vs 4-bit Attention Patterns")
    fig.tight_layout()
    path = os.path.join(save_dir, "attention_jsd_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 2. Most and least affected heads
    flat_indices = np.argsort(jsd_matrix.ravel())
    n_show = 10

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Least affected (smallest JSD)
    least = flat_indices[:n_show]
    least_labels = [f"L{i // jsd_matrix.shape[1]}H{i % jsd_matrix.shape[1]}" for i in least]
    least_vals = [jsd_matrix.ravel()[i] for i in least]
    axes[0].barh(least_labels, least_vals, color=sns.color_palette("Greens_r", n_show))
    axes[0].set_xlabel("JSD")
    axes[0].set_title("10 Least Affected Heads (best preserved)")
    axes[0].invert_yaxis()

    # Most affected (largest JSD)
    most = flat_indices[-n_show:][::-1]
    most_labels = [f"L{i // jsd_matrix.shape[1]}H{i % jsd_matrix.shape[1]}" for i in most]
    most_vals = [jsd_matrix.ravel()[i] for i in most]
    axes[1].barh(most_labels, most_vals, color=sns.color_palette("Reds", n_show))
    axes[1].set_xlabel("JSD")
    axes[1].set_title("10 Most Affected Heads (worst preserved)")
    axes[1].invert_yaxis()

    fig.suptitle("Attention Head Sensitivity to 4-bit Quantization", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "attention_most_least_affected.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 3. Previous-token head scores: FP vs Q
    fig, ax = plt.subplots(figsize=(12, 5))
    fp_flat = prev_scores_fp.ravel()
    q_flat = prev_scores_q.ravel()
    labels = [f"L{l}H{h}" for l in range(prev_scores_fp.shape[0]) for h in range(prev_scores_fp.shape[1])]

    # Only show heads with high previous-token score in FP
    threshold = np.percentile(fp_flat, 90)
    mask = fp_flat >= threshold
    selected_labels = [l for l, m in zip(labels, mask) if m]
    selected_fp = fp_flat[mask]
    selected_q = q_flat[mask]

    x = np.arange(len(selected_labels))
    width = 0.35
    ax.bar(x - width / 2, selected_fp, width, label="FP16", color="#4C72B0")
    ax.bar(x + width / 2, selected_q, width, label="4-bit", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(selected_labels, rotation=45, ha="right")
    ax.set_ylabel("Previous-Token Attention Score")
    ax.set_title("Previous-Token Heads: FP vs 4-bit (top 10% by FP score)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(save_dir, "attention_prev_token_heads.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 4. Induction head region focus (middle layers)
    n_l = jsd_matrix.shape[0]
    mid = n_l // 2
    reg_start = max(0, mid - 1)
    reg_end = min(n_l, mid + 2)
    fig, ax = plt.subplots(figsize=(max(8, n_heads * 0.7), 4))
    induction_jsd = jsd_matrix[reg_start:reg_end, :]
    sns.heatmap(
        induction_jsd,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        xticklabels=[f"H{i}" for i in range(induction_jsd.shape[1])],
        yticklabels=[f"L{i}" for i in range(reg_start, reg_end)],
        ax=ax,
    )
    ax.set_title(f"Induction Head Region (Layers {reg_start}-{reg_end-1}): JSD under Quantization")
    fig.tight_layout()
    path = os.path.join(save_dir, "attention_induction_region.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_attention_analysis(
    fp_model=None,
    q_model=None,
    tokens=None,
    save_dir: str = "results",
    n_prompts: int = 50,
    batch_size: int = 8,
) -> dict:
    """Full attention pattern analysis pipeline."""
    if fp_model is None or q_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp_model, q_model = load_models(device=device)

    if tokens is None:
        tokens = tokenize_prompts(fp_model)

    n_layers = fp_model.cfg.n_layers
    n_heads = fp_model.cfg.n_heads

    print("\n=== Extracting FP attention patterns ===")
    fp_patterns = extract_attention_patterns(fp_model, tokens, n_prompts=n_prompts, batch_size=batch_size)

    print("\n=== Extracting quantized attention patterns ===")
    q_patterns = extract_attention_patterns(q_model, tokens, n_prompts=n_prompts, batch_size=batch_size)

    print("\n=== Computing JSD per head ===")
    jsd_matrix = compute_head_jsd(fp_patterns, q_patterns, n_layers=n_layers, n_heads=n_heads)

    print("\n=== Computing previous-token scores ===")
    prev_scores_fp = compute_prev_token_score(fp_patterns, n_layers=n_layers, n_heads=n_heads)
    prev_scores_q = compute_prev_token_score(q_patterns, n_layers=n_layers, n_heads=n_heads)

    # Print summary
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Mean JSD across all heads: {jsd_matrix.mean():.6f}")
    print(f"Max  JSD: {jsd_matrix.max():.6f} at L{np.unravel_index(jsd_matrix.argmax(), jsd_matrix.shape)[0]}H{np.unravel_index(jsd_matrix.argmax(), jsd_matrix.shape)[1]}")
    print(f"Min  JSD: {jsd_matrix.min():.6f} at L{np.unravel_index(jsd_matrix.argmin(), jsd_matrix.shape)[0]}H{np.unravel_index(jsd_matrix.argmin(), jsd_matrix.shape)[1]}")

    # Induction head region (middle layers)
    mid = n_layers // 2
    reg_start = max(0, mid - 1)
    reg_end = min(n_layers, mid + 2)
    induction_jsd = jsd_matrix[reg_start:reg_end, :].mean()
    other_mask = np.ones(n_layers, dtype=bool)
    other_mask[reg_start:reg_end] = False
    other_jsd = jsd_matrix[other_mask].mean() if other_mask.any() else 0.0
    print(f"\nInduction region (L{reg_start}-{reg_end-1}) mean JSD: {induction_jsd:.6f}")
    print(f"Other layers mean JSD:            {other_jsd:.6f}")

    # Previous-token head preservation
    top_prev = np.argsort(prev_scores_fp.ravel())[-5:][::-1]
    print("\nTop previous-token heads (FP score -> Q score):")
    for idx in top_prev:
        l, h = divmod(idx, n_heads)
        print(f"  L{l}H{h}: {prev_scores_fp[l, h]:.4f} -> {prev_scores_q[l, h]:.4f} (delta: {abs(prev_scores_fp[l, h] - prev_scores_q[l, h]):.4f})")

    plot_attention_analysis(jsd_matrix, prev_scores_fp, prev_scores_q, save_dir=save_dir)

    del fp_patterns, q_patterns

    return {
        "jsd_matrix": jsd_matrix,
        "prev_scores_fp": prev_scores_fp,
        "prev_scores_q": prev_scores_q,
    }


if __name__ == "__main__":
    run_attention_analysis()
