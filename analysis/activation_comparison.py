"""
Activation comparison between FP16 and 4-bit quantized GPT-2.

For each layer (0-11) and component (residual stream, MLP output, attention output):
  - Extract activations on 200 prompts for both models
  - Compute cosine similarity, L2 distance, Pearson correlation
  - Generate layer-by-layer degradation plots
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
from utils.metrics import cosine_similarity_batch, l2_distance_batch, pearson_correlation_batch


# ---------------------------------------------------------------------------
# Hook names for each component at each layer
# ---------------------------------------------------------------------------

def get_hook_names(n_layers: int = 12) -> dict[str, list[str]]:
    """Return TransformerLens hook point names for each component type."""
    return {
        "resid_post": [f"blocks.{i}.hook_resid_post" for i in range(n_layers)],
        "mlp_out":    [f"blocks.{i}.hook_mlp_out" for i in range(n_layers)],
        "attn_out":   [f"blocks.{i}.hook_attn_out" for i in range(n_layers)],
    }


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_activations(
    model,
    tokens: torch.Tensor,
    hook_names: list[str],
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Run prompts through model and collect activations at specified hook points.

    Args:
        model: HookedTransformer.
        tokens: Input token IDs, shape (n_prompts, seq_len).
        hook_names: List of TransformerLens hook point names.
        batch_size: Process prompts in batches to fit in memory.

    Returns:
        Dict mapping hook_name -> tensor of shape (n_prompts, seq_len, d_model).
    """
    device = next(model.parameters()).device
    all_activations = {name: [] for name in hook_names}

    for start in tqdm(range(0, len(tokens), batch_size), desc="Extracting activations"):
        batch = tokens[start : start + batch_size].to(device)
        _, cache = model.run_with_cache(batch, names_filter=hook_names)

        for name in hook_names:
            all_activations[name].append(cache[name].cpu())

    return {name: torch.cat(acts, dim=0) for name, acts in all_activations.items()}


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_activations(
    fp_acts: dict[str, torch.Tensor],
    q_acts: dict[str, torch.Tensor],
    component_hooks: dict[str, list[str]],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute per-layer metrics between FP and quantized activations.

    Args:
        fp_acts: FP model activations (hook_name -> tensor).
        q_acts: Quantized model activations.
        component_hooks: Dict mapping component name -> list of hook names per layer.

    Returns:
        Nested dict: component -> metric -> np.ndarray of shape (n_layers,).
    """
    results = {}

    for comp_name, hooks in component_hooks.items():
        n_layers = len(hooks)
        cos_sims = np.zeros(n_layers)
        l2_dists = np.zeros(n_layers)
        correlations = np.zeros(n_layers)

        for layer_idx, hook_name in enumerate(hooks):
            fp = fp_acts[hook_name]
            q = q_acts[hook_name]

            cos = cosine_similarity_batch(fp, q).mean().item()
            l2 = l2_distance_batch(fp, q).mean().item()
            corr = pearson_correlation_batch(fp, q).mean().item()

            cos_sims[layer_idx] = cos
            l2_dists[layer_idx] = l2
            correlations[layer_idx] = corr

        results[comp_name] = {
            "cosine_similarity": cos_sims,
            "l2_distance": l2_dists,
            "pearson_correlation": correlations,
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_activation_comparison(
    results: dict[str, dict[str, np.ndarray]],
    save_dir: str = "results",
):
    """
    Generate layer-by-layer degradation plots for each metric.

    Creates three plots:
      1. Cosine similarity per layer (higher = better preserved)
      2. L2 distance per layer (lower = better preserved)
      3. Pearson correlation per layer (higher = better preserved)
    """
    os.makedirs(save_dir, exist_ok=True)

    components = list(results.keys())
    n_layers = len(next(iter(results.values()))["cosine_similarity"])
    layers = np.arange(n_layers)

    sns.set_theme(style="whitegrid", font_scale=1.2)
    colors = sns.color_palette("husl", len(components))

    metrics_config = [
        ("cosine_similarity", "Cosine Similarity (FP vs 4-bit)", "higher = better preserved", True),
        ("l2_distance", "L2 Distance (FP vs 4-bit)", "lower = better preserved", False),
        ("pearson_correlation", "Pearson Correlation (FP vs 4-bit)", "higher = better preserved", True),
    ]

    for metric_key, title, ylabel_note, higher_better in metrics_config:
        fig, ax = plt.subplots(figsize=(12, 6))

        for comp_name, color in zip(components, colors):
            values = results[comp_name][metric_key]
            label = comp_name.replace("_", " ").title()
            ax.plot(layers, values, "o-", color=color, label=label, linewidth=2, markersize=6)

        ax.set_xlabel("Layer")
        ax.set_ylabel(f"{metric_key.replace('_', ' ').title()}\n({ylabel_note})")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        path = os.path.join(save_dir, f"activation_{metric_key}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

    # Combined summary plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax, (metric_key, title, ylabel_note, _) in zip(axes, metrics_config):
        for comp_name, color in zip(components, colors):
            values = results[comp_name][metric_key]
            label = comp_name.replace("_", " ").title()
            ax.plot(layers, values, "o-", color=color, label=label, linewidth=2, markersize=5)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_key.replace("_", " ").title())
        ax.set_title(title.split("(")[0].strip())
        ax.set_xticks(layers)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Impact of 4-bit Quantization on Activations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "activation_comparison_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_activation_comparison(
    fp_model=None,
    q_model=None,
    tokens=None,
    save_dir: str = "results",
    batch_size: int = 8,
) -> dict:
    """
    Full activation comparison pipeline.

    If models/tokens not provided, loads them from scratch.
    Returns the results dict for downstream use.
    """
    if fp_model is None or q_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp_model, q_model = load_models(device=device)

    if tokens is None:
        tokens = tokenize_prompts(fp_model)

    n_layers = fp_model.cfg.n_layers
    component_hooks = get_hook_names(n_layers)

    # Flatten all hook names for extraction
    all_hooks = []
    for hooks in component_hooks.values():
        all_hooks.extend(hooks)

    print("\n=== Extracting FP model activations ===")
    fp_acts = extract_activations(fp_model, tokens, all_hooks, batch_size=batch_size)

    print("\n=== Extracting quantized model activations ===")
    q_acts = extract_activations(q_model, tokens, all_hooks, batch_size=batch_size)

    print("\n=== Computing comparison metrics ===")
    results = compare_activations(fp_acts, q_acts, component_hooks)

    # Print summary
    print("\n" + "=" * 60)
    print("ACTIVATION COMPARISON SUMMARY")
    print("=" * 60)
    for comp_name, metrics in results.items():
        cos = metrics["cosine_similarity"]
        print(f"\n{comp_name}:")
        print(f"  Cosine sim  — mean: {cos.mean():.4f}, min: {cos.min():.4f} (layer {cos.argmin()}), max: {cos.max():.4f} (layer {cos.argmax()})")
        print(f"  L2 distance — mean: {metrics['l2_distance'].mean():.4f}")
        print(f"  Correlation — mean: {metrics['pearson_correlation'].mean():.4f}")

    plot_activation_comparison(results, save_dir=save_dir)

    # Free memory
    del fp_acts, q_acts

    return results


if __name__ == "__main__":
    run_activation_comparison()
