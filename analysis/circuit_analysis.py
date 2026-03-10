"""
Circuit analysis: identify induction heads and test if they survive quantization.

Induction heads implement the pattern [A][B]...[A] -> predict [B].
They are a key circuit in transformer language models, typically found in layers 5-6.

This module:
  1. Generates random repeated sequences to trigger induction behavior
  2. Scores each head on how well it performs the induction pattern
  3. Compares scores between FP and quantized models
"""

import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_models


# ---------------------------------------------------------------------------
# Induction sequence generation
# ---------------------------------------------------------------------------

def generate_induction_sequences(
    vocab_size: int = 50257,
    bos_token_id: int = 50256,
    n_sequences: int = 50,
    half_len: int = 25,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate repeated random token sequences to test induction heads.

    Each sequence has the form [BOS] [A1 A2 ... An] [A1 A2 ... An]
    where the second half repeats the first. Induction heads should attend
    from the second occurrence of Ai to the token after the first occurrence of Ai.

    Args:
        vocab_size: Vocabulary size.
        bos_token_id: BOS token ID (50256 for GPT-2, 0 for Pythia).
        n_sequences: Number of test sequences.
        half_len: Length of each half-sequence.
        seed: Random seed.

    Returns:
        tokens: Shape (n_sequences, 2 * half_len + 1) with BOS prepended.
    """
    rng = torch.Generator().manual_seed(seed)

    sequences = []
    for _ in range(n_sequences):
        # Random tokens, avoiding very low IDs (special tokens)
        half = torch.randint(100, min(vocab_size - 100, vocab_size), (half_len,), generator=rng)
        full = torch.cat([
            torch.tensor([bos_token_id]),
            half,
            half,  # Repeat
        ])
        sequences.append(full)

    return torch.stack(sequences)


# ---------------------------------------------------------------------------
# Induction score computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_induction_scores(
    model,
    sequences: torch.Tensor,
    batch_size: int = 10,
) -> np.ndarray:
    """
    Compute induction score for each attention head.

    For a repeated sequence [BOS][A1..An][A1..An], an induction head at position
    (half_len + i + 1) should attend to position (i + 1) — i.e., the token after
    the first occurrence of the current token.

    The induction score for a head = mean attention weight on the "induction target"
    position across all repeated-sequence positions.

    Args:
        model: HookedTransformer.
        sequences: Shape (n_sequences, seq_len) with repeated halves.

    Returns:
        scores: Shape (n_layers, n_heads). Higher = more induction-like.
    """
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    half_len = (sequences.shape[1] - 1) // 2

    hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]
    scores = np.zeros((n_layers, n_heads))
    count = 0

    for start in tqdm(range(0, len(sequences), batch_size), desc="Induction scores"):
        batch = sequences[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)

        for layer_idx in range(n_layers):
            hook = f"blocks.{layer_idx}.attn.hook_pattern"
            pattern = cache[hook]  # (batch, n_heads, seq_len, seq_len)

            for head_idx in range(n_heads):
                head_pat = pattern[:, head_idx]  # (batch, seq_len, seq_len)

                # For each position in the second half (index half_len+1 to 2*half_len),
                # the induction target is the position after the first occurrence:
                # position (i - half_len + 1) for query position i
                induction_attn = 0.0
                n_positions = 0
                for pos in range(half_len + 1, 2 * half_len + 1):
                    target_pos = pos - half_len + 1  # Position after first occurrence
                    if target_pos < head_pat.shape[-1]:
                        induction_attn += head_pat[:, pos, target_pos].mean().item()
                        n_positions += 1

                if n_positions > 0:
                    scores[layer_idx, head_idx] += induction_attn / n_positions

        count += 1

    scores /= max(count, 1)
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_circuit_analysis(
    fp_scores: np.ndarray,
    q_scores: np.ndarray,
    save_dir: str = "results",
):
    """Generate circuit analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="white", font_scale=1.1)

    n_layers, n_heads = fp_scores.shape

    # 1. Induction score heatmaps side by side
    show_annot = n_layers * n_heads <= 200
    fig_w = max(18, n_heads * 1.4)
    fig_h = max(7, n_layers * 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    vmin = min(fp_scores.min(), q_scores.min())
    vmax = max(fp_scores.max(), q_scores.max())

    for ax, scores, title in [
        (axes[0], fp_scores, "FP Model — Induction Scores"),
        (axes[1], q_scores, "4-bit Quantized — Induction Scores"),
    ]:
        sns.heatmap(
            scores,
            annot=show_annot,
            fmt=".3f" if show_annot else "",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            xticklabels=[f"H{i}" for i in range(n_heads)],
            yticklabels=[f"L{i}" for i in range(n_layers)],
            ax=ax,
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)

    fig.suptitle("Induction Head Detection: FP vs 4-bit Quantized", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "circuit_induction_heatmaps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 2. Score difference heatmap
    diff = fp_scores - q_scores
    fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.8), max(7, n_layers * 0.5)))
    sns.heatmap(
        diff,
        annot=show_annot,
        fmt=".3f" if show_annot else "",
        cmap="RdBu_r",
        center=0,
        xticklabels=[f"H{i}" for i in range(n_heads)],
        yticklabels=[f"L{i}" for i in range(n_layers)],
        ax=ax,
    )
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Induction Score Difference (FP - 4bit)\nPositive = FP stronger, Negative = 4bit stronger")
    fig.tight_layout()
    path = os.path.join(save_dir, "circuit_induction_diff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 3. Scatter: FP score vs Q score for each head
    fig, ax = plt.subplots(figsize=(8, 8))
    fp_flat = fp_scores.ravel()
    q_flat = q_scores.ravel()

    # Color by layer
    colors = plt.cm.viridis(np.repeat(np.linspace(0, 1, n_layers), n_heads))

    ax.scatter(fp_flat, q_flat, c=colors, alpha=0.7, s=50, edgecolors="black", linewidths=0.5)
    ax.plot([0, max(fp_flat.max(), q_flat.max())],
            [0, max(fp_flat.max(), q_flat.max())],
            "k--", alpha=0.5, label="y = x (perfect preservation)")

    # Annotate top induction heads
    threshold = np.percentile(fp_flat, 95)
    for idx in np.where(fp_flat >= threshold)[0]:
        l, h = divmod(idx, n_heads)
        ax.annotate(f"L{l}H{h}", (fp_flat[idx], q_flat[idx]),
                     fontsize=8, ha="left", va="bottom")

    ax.set_xlabel("FP Induction Score")
    ax.set_ylabel("4-bit Induction Score")
    ax.set_title("Induction Head Preservation under Quantization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar for layer
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, n_layers - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Layer")

    fig.tight_layout()
    path = os.path.join(save_dir, "circuit_induction_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 4. Top induction heads comparison bar chart
    # Find top 10 heads by FP score
    top_idx = np.argsort(fp_flat)[-10:][::-1]
    labels = [f"L{idx // n_heads}H{idx % n_heads}" for idx in top_idx]
    fp_vals = fp_flat[top_idx]
    q_vals = q_flat[top_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, fp_vals, width, label="FP", color="#4C72B0")
    ax.bar(x + width / 2, q_vals, width, label="4-bit", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Induction Score")
    ax.set_title("Top 10 Induction Heads: FP vs 4-bit")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(save_dir, "circuit_top_induction_heads.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_circuit_analysis(
    fp_model=None,
    q_model=None,
    save_dir: str = "results",
    n_sequences: int = 50,
    batch_size: int = 10,
) -> dict:
    """Full circuit analysis pipeline."""
    if fp_model is None or q_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp_model, q_model = load_models(device=device)

    # Generate induction test sequences with model-specific BOS token
    print("\n=== Generating induction test sequences ===")
    vocab_size = fp_model.cfg.d_vocab
    bos_id = fp_model.tokenizer.bos_token_id
    if bos_id is None:
        bos_id = 0
    sequences = generate_induction_sequences(
        vocab_size=vocab_size, bos_token_id=bos_id, n_sequences=n_sequences
    )
    print(f"Generated {len(sequences)} sequences of length {sequences.shape[1]}")

    print("\n=== Computing FP induction scores ===")
    fp_scores = compute_induction_scores(fp_model, sequences, batch_size=batch_size)

    print("\n=== Computing 4-bit induction scores ===")
    q_scores = compute_induction_scores(q_model, sequences, batch_size=batch_size)

    # Summary
    n_layers, n_heads = fp_scores.shape
    print("\n" + "=" * 60)
    print("CIRCUIT ANALYSIS SUMMARY")
    print("=" * 60)

    # Identify induction heads (score > 2x mean)
    fp_threshold = fp_scores.mean() + 2 * fp_scores.std()
    fp_induction = np.argwhere(fp_scores > fp_threshold)
    print(f"\nInduction heads in FP model (score > {fp_threshold:.3f}):")
    for l, h in fp_induction:
        preserved = "PRESERVED" if q_scores[l, h] > fp_threshold * 0.5 else "DEGRADED"
        print(f"  L{l}H{h}: FP={fp_scores[l, h]:.4f}, Q={q_scores[l, h]:.4f} [{preserved}]")

    # Correlation between FP and Q scores
    corr = np.corrcoef(fp_scores.ravel(), q_scores.ravel())[0, 1]
    print(f"\nOverall score correlation (FP vs Q): {corr:.4f}")

    # Layer-wise summary
    print("\nPer-layer mean induction score:")
    for l in range(n_layers):
        fp_mean = fp_scores[l].mean()
        q_mean = q_scores[l].mean()
        print(f"  L{l}: FP={fp_mean:.4f}, Q={q_mean:.4f}, ratio={q_mean / max(fp_mean, 1e-8):.2f}")

    # Functional test: do induction heads still work?
    n_survived = sum(1 for l, h in fp_induction if q_scores[l, h] > fp_threshold * 0.5)
    n_total = len(fp_induction)
    survival_rate = n_survived / max(n_total, 1)
    print(f"\nInduction head survival rate: {n_survived}/{n_total} = {survival_rate:.0%}")

    plot_circuit_analysis(fp_scores, q_scores, save_dir=save_dir)

    return {
        "fp_scores": fp_scores,
        "q_scores": q_scores,
        "induction_heads": fp_induction,
        "survival_rate": survival_rate,
        "correlation": corr,
    }


if __name__ == "__main__":
    run_circuit_analysis()
