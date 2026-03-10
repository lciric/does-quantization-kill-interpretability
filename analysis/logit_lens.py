"""
Logit lens analysis: how token predictions evolve layer by layer.

At each layer, projects the residual stream onto the vocabulary via the
unembedding matrix to see what the model "thinks" at intermediate stages.
Compares FP and 4-bit quantized models to find where predictions diverge.
"""

import sys
import os

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_models, LOGIT_LENS_PROMPTS
from utils.metrics import kl_divergence, top_k_agreement


# ---------------------------------------------------------------------------
# Logit lens computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_logit_lens(
    model,
    prompt: str,
    target_token: str,
) -> dict:
    """
    Apply logit lens to a single prompt.

    For each layer, project the residual stream at the last token position
    through the unembedding matrix to get vocabulary logits.

    Args:
        model: HookedTransformer.
        prompt: Input text.
        target_token: Expected next token (e.g., " Paris").

    Returns:
        Dict with keys:
            - layer_probs: (n_layers+1, vocab_size) — probability at each layer
            - target_probs: (n_layers+1,) — probability of target token at each layer
            - top5_tokens: list of (n_layers+1) lists of top-5 token strings
            - top5_probs: (n_layers+1, 5) — probabilities of top-5 tokens
    """
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers

    # Tokenize
    tokens = model.to_tokens(prompt).to(device)
    target_id = model.to_tokens(target_token, prepend_bos=False)[0, 0].item()

    # Hook names: residual stream after each layer + after embedding
    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    # Unembedding matrix
    W_U = model.W_U  # (d_model, vocab_size)
    b_U = model.b_U if hasattr(model, "b_U") and model.b_U is not None else 0

    layer_probs = []
    target_probs = []
    top5_tokens = []
    top5_probs_list = []

    # Also apply to the embedding output (layer -1 equivalent: resid_pre of block 0)
    # We use hook_resid_post for layers 0..n_layers-1
    # For the "pre-layer-0" view, use blocks.0.hook_resid_pre if available

    for layer_idx in range(n_layers):
        hook = f"blocks.{layer_idx}.hook_resid_post"
        resid = cache[hook][0, -1, :]  # Last token position, shape (d_model,)

        # Apply layer norm if the model uses it before unembedding
        if model.cfg.normalization_type is not None:
            resid = model.ln_final(resid.unsqueeze(0)).squeeze(0)

        logits = resid @ W_U + b_U  # (vocab_size,)
        probs = F.softmax(logits.float(), dim=-1)

        layer_probs.append(probs.cpu())
        target_probs.append(probs[target_id].item())

        top5 = probs.topk(5)
        top5_tokens.append([model.tokenizer.decode([idx]) for idx in top5.indices.tolist()])
        top5_probs_list.append(top5.values.cpu().numpy())

    return {
        "layer_probs": torch.stack(layer_probs),  # (n_layers, vocab_size)
        "target_probs": np.array(target_probs),  # (n_layers,)
        "top5_tokens": top5_tokens,
        "top5_probs": np.array(top5_probs_list),  # (n_layers, 5)
        "target_token": target_token,
        "target_id": target_id,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_logit_lens(
    fp_result: dict,
    q_result: dict,
) -> dict:
    """
    Compare logit lens results between FP and quantized models.

    Returns:
        Dict with per-layer KL divergence, target prob difference, top-5 agreement.
    """
    n_layers = len(fp_result["target_probs"])

    kl_per_layer = np.zeros(n_layers)
    target_diff = np.zeros(n_layers)
    top5_agree = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        fp_probs = fp_result["layer_probs"][layer_idx].unsqueeze(0)
        q_probs = q_result["layer_probs"][layer_idx].unsqueeze(0)

        # KL from log-probs
        kl = (fp_probs * (fp_probs.clamp(min=1e-10).log() - q_probs.clamp(min=1e-10).log())).sum().item()
        kl_per_layer[layer_idx] = max(kl, 0)

        target_diff[layer_idx] = fp_result["target_probs"][layer_idx] - q_result["target_probs"][layer_idx]

        # Top-5 agreement
        fp_top5 = set(fp_result["top5_tokens"][layer_idx])
        q_top5 = set(q_result["top5_tokens"][layer_idx])
        top5_agree[layer_idx] = len(fp_top5 & q_top5) / 5

    return {
        "kl_divergence": kl_per_layer,
        "target_prob_diff": target_diff,
        "top5_agreement": top5_agree,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_logit_lens(
    all_fp_results: list[dict],
    all_q_results: list[dict],
    all_comparisons: list[dict],
    prompts: list[tuple[str, str]],
    save_dir: str = "results",
):
    """Generate logit lens visualization plots."""
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    n_prompts = len(prompts)
    n_layers = len(all_fp_results[0]["target_probs"])
    layers = np.arange(n_layers)

    # 1. Target token probability at each layer — FP vs Q
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
    axes = axes.ravel()

    for i, (ax, (prompt, target)) in enumerate(zip(axes, prompts)):
        fp_probs = all_fp_results[i]["target_probs"]
        q_probs = all_q_results[i]["target_probs"]

        ax.plot(layers, fp_probs, "o-", color="#4C72B0", label="FP", linewidth=2)
        ax.plot(layers, q_probs, "s--", color="#DD8452", label="4-bit", linewidth=2)
        ax.set_xlabel("Layer")
        if i % 5 == 0:
            ax.set_ylabel(f"P('{target.strip()}')")
        short_prompt = prompt[:30] + "..." if len(prompt) > 30 else prompt
        ax.set_title(f'"{short_prompt}"', fontsize=10)
        ax.set_xticks(layers)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Logit Lens: Target Token Probability by Layer (FP vs 4-bit)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(save_dir, "logit_lens_target_probs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 2. KL divergence heatmap across prompts and layers
    kl_matrix = np.array([comp["kl_divergence"] for comp in all_comparisons])  # (n_prompts, n_layers)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        kl_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=[f"L{i}" for i in range(n_layers)],
        yticklabels=[f'"{p[:25]}..."' for p, _ in prompts],
        ax=ax,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Prompt")
    ax.set_title("KL Divergence (FP vs 4-bit) at Each Layer")
    fig.tight_layout()
    path = os.path.join(save_dir, "logit_lens_kl_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 3. Average divergence curve
    mean_kl = kl_matrix.mean(axis=0)
    mean_agree = np.array([comp["top5_agreement"] for comp in all_comparisons]).mean(axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1 = "#e74c3c"
    color2 = "#2ecc71"

    ax1.plot(layers, mean_kl, "o-", color=color1, linewidth=2, label="KL Divergence")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean KL Divergence", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    ax2.plot(layers, mean_agree, "s--", color=color2, linewidth=2, label="Top-5 Agreement")
    ax2.set_ylabel("Top-5 Token Agreement", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
    ax1.set_title("Logit Lens: Where Does Quantization Diverge?")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "logit_lens_divergence_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # 4. Top-5 token comparison table for one prompt
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")

    # Pick prompt 0 as example
    fp_r = all_fp_results[0]
    q_r = all_q_results[0]
    cell_text = []
    for layer_idx in range(n_layers):
        fp_top = ", ".join([f"{t}({p:.2f})" for t, p in zip(fp_r["top5_tokens"][layer_idx], fp_r["top5_probs"][layer_idx])])
        q_top = ", ".join([f"{t}({p:.2f})" for t, p in zip(q_r["top5_tokens"][layer_idx], q_r["top5_probs"][layer_idx])])
        cell_text.append([f"L{layer_idx}", fp_top, q_top])

    table = ax.table(
        cellText=cell_text,
        colLabels=["Layer", "FP Top-5", "4-bit Top-5"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    prompt_str = prompts[0][0]
    ax.set_title(f'Logit Lens: "{prompt_str}" -> "{prompts[0][1].strip()}"', fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(save_dir, "logit_lens_token_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_logit_lens(
    fp_model=None,
    q_model=None,
    save_dir: str = "results",
) -> dict:
    """Full logit lens analysis pipeline."""
    if fp_model is None or q_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp_model, q_model = load_models(device=device)

    prompts = LOGIT_LENS_PROMPTS

    all_fp_results = []
    all_q_results = []
    all_comparisons = []

    print("\n=== Running Logit Lens Analysis ===")
    for prompt, target in tqdm(prompts, desc="Logit lens"):
        fp_result = apply_logit_lens(fp_model, prompt, target)
        q_result = apply_logit_lens(q_model, prompt, target)
        comparison = compare_logit_lens(fp_result, q_result)

        all_fp_results.append(fp_result)
        all_q_results.append(q_result)
        all_comparisons.append(comparison)

    # Summary
    print("\n" + "=" * 60)
    print("LOGIT LENS SUMMARY")
    print("=" * 60)

    kl_matrix = np.array([c["kl_divergence"] for c in all_comparisons])
    agree_matrix = np.array([c["top5_agreement"] for c in all_comparisons])

    mean_kl = kl_matrix.mean(axis=0)
    diverge_layer = np.argmax(mean_kl > mean_kl.mean())
    print(f"Mean KL per layer: {', '.join([f'L{i}:{v:.4f}' for i, v in enumerate(mean_kl)])}")
    print(f"Divergence starts becoming significant around layer {diverge_layer}")
    print(f"Mean top-5 agreement: {agree_matrix.mean():.3f}")

    for i, (prompt, target) in enumerate(prompts):
        fp_final = all_fp_results[i]["target_probs"][-1]
        q_final = all_q_results[i]["target_probs"][-1]
        print(f'  "{prompt[:40]}" -> "{target}": FP={fp_final:.3f}, Q={q_final:.3f}')

    plot_logit_lens(all_fp_results, all_q_results, all_comparisons, prompts, save_dir=save_dir)

    return {
        "fp_results": all_fp_results,
        "q_results": all_q_results,
        "comparisons": all_comparisons,
    }


if __name__ == "__main__":
    run_logit_lens()
