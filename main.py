"""
P2 Mechanistic Interpretability Pipeline — Scaling Study.

Runs 5 analyses comparing FP models against quantized variants (RTN and GPTQ)
across multiple model families and sizes.

Supported models: GPT-2, GPT-2-XL, Pythia-410M, Pythia-1.4B, Pythia-2.8B

Usage:
    python main.py --model gpt2                     # Single model (backward compat)
    python main.py --models-all                      # All 5 models
    python main.py --model EleutherAI/pythia-410m    # Specific model
"""

import argparse
import gc
import os
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.model_loader import (
    load_fp_model, create_quantized_model, create_gptq_quantized_model,
    tokenize_prompts,
)
from analysis.activation_comparison import run_activation_comparison
from analysis.attention_patterns import run_attention_analysis
from analysis.feature_analysis import run_feature_analysis
from analysis.logit_lens import run_logit_lens
from analysis.circuit_analysis import run_circuit_analysis


MODELS_ALL = [
    "gpt2",
    "gpt2-xl",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
]


def model_short_name(name: str) -> str:
    """Extract short name for directory paths: 'EleutherAI/pythia-410m' -> 'pythia-410m'."""
    return name.split("/")[-1]


# ---------------------------------------------------------------------------
# Per-variant analysis runner
# ---------------------------------------------------------------------------

def run_analyses(label, fp_model, q_model, tokens, save_dir, args, trained_sae=None):
    """Run all 5 analyses for one FP vs quantized pair. Returns (results, sae)."""
    results = {}
    n_layers = fp_model.cfg.n_layers
    d_model = fp_model.cfg.d_model

    # SAE parameters adapted to model size
    sae_layer = n_layers // 2
    if args.sae_epochs is not None:
        sae_epochs = args.sae_epochs
    else:
        sae_epochs = 1000 if d_model <= 768 else 500

    print(f"\n{'=' * 70}")
    print(f"ANALYZING: FP vs {label.upper()} 4-bit")
    print(f"{'=' * 70}")

    # 1. Activation comparison
    if "activation" not in args.skip:
        print(f"\n{'#' * 70}")
        print(f"# {label.upper()} — 1. ACTIVATION COMPARISON")
        print(f"{'#' * 70}")
        t0 = time.time()
        results["activation"] = run_activation_comparison(
            fp_model, q_model, tokens,
            save_dir=save_dir, batch_size=args.batch_size,
        )
        print(f"Completed in {time.time() - t0:.1f}s")

    # 2. Attention patterns
    if "attention" not in args.skip:
        print(f"\n{'#' * 70}")
        print(f"# {label.upper()} — 2. ATTENTION PATTERN ANALYSIS")
        print(f"{'#' * 70}")
        t0 = time.time()
        results["attention"] = run_attention_analysis(
            fp_model, q_model, tokens,
            save_dir=save_dir, n_prompts=50, batch_size=args.batch_size,
        )
        print(f"Completed in {time.time() - t0:.1f}s")

    # 3. Feature analysis (SAE — train once, reuse)
    if "feature" not in args.skip:
        print(f"\n{'#' * 70}")
        print(f"# {label.upper()} — 3. FEATURE ANALYSIS (SPARSE AUTOENCODER)")
        print(f"{'#' * 70}")
        t0 = time.time()
        results["feature"] = run_feature_analysis(
            fp_model, q_model, tokens,
            save_dir=save_dir, layer=sae_layer,
            n_epochs=sae_epochs, batch_size=args.batch_size,
            sae=trained_sae,
        )
        if trained_sae is None:
            trained_sae = results["feature"]["sae"]
        print(f"Completed in {time.time() - t0:.1f}s")

    # 4. Logit lens
    if "logit" not in args.skip:
        print(f"\n{'#' * 70}")
        print(f"# {label.upper()} — 4. LOGIT LENS")
        print(f"{'#' * 70}")
        t0 = time.time()
        results["logit"] = run_logit_lens(
            fp_model, q_model,
            save_dir=save_dir,
        )
        print(f"Completed in {time.time() - t0:.1f}s")

    # 5. Circuit analysis
    if "circuit" not in args.skip:
        print(f"\n{'#' * 70}")
        print(f"# {label.upper()} — 5. CIRCUIT ANALYSIS (INDUCTION HEADS)")
        print(f"{'#' * 70}")
        t0 = time.time()
        results["circuit"] = run_circuit_analysis(
            fp_model, q_model,
            save_dir=save_dir, n_sequences=50, batch_size=args.batch_size,
        )
        print(f"Completed in {time.time() - t0:.1f}s")

    return results, trained_sae


# ---------------------------------------------------------------------------
# Per-model comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(all_results, save_dir, short_name=""):
    """Side-by-side bar chart comparing RTN vs GPTQ across key metrics."""
    labels = list(all_results.keys())

    metric_names = []
    metric_values = {label: [] for label in labels}

    if all("activation" in r for r in all_results.values()):
        metric_names.append("Resid\nCos Sim")
        for label in labels:
            val = all_results[label]["activation"]["resid_post"]["cosine_similarity"].mean()
            metric_values[label].append(float(val))

    if all("attention" in r for r in all_results.values()):
        metric_names.append("Attn JSD\n(x100)")
        for label in labels:
            val = all_results[label]["attention"]["jsd_matrix"].mean() * 100
            metric_values[label].append(float(val))

    if all("feature" in r for r in all_results.values()):
        metric_names.append("Feature\nSurvival")
        for label in labels:
            metric_values[label].append(float(all_results[label]["feature"]["survival_rate"]))

    if all("circuit" in r for r in all_results.values()):
        metric_names.append("Induction\nSurvival")
        for label in labels:
            metric_values[label].append(float(all_results[label]["circuit"]["survival_rate"]))

        metric_names.append("Induction\nCorrelation")
        for label in labels:
            metric_values[label].append(float(all_results[label]["circuit"]["correlation"]))

    if not metric_names:
        return

    x = np.arange(len(metric_names))
    width = 0.35
    colors = {"rtn": "#e74c3c", "gptq": "#2ecc71"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(
            x + offset, metric_values[label], width,
            label=f"{label.upper()} 4-bit", color=colors.get(label, f"C{i}"),
        )
        for bar, val in zip(bars, metric_values[label]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    title = f"RTN vs GPTQ: Quantization Impact on {short_name}" if short_name else "RTN vs GPTQ"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(save_dir, "comparison_rtn_vs_gptq.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Single-model pipeline
# ---------------------------------------------------------------------------

def run_single_model(model_name, args, scaling_results):
    """Run full pipeline for one model. Updates scaling_results in-place."""
    short = model_short_name(model_name)
    model_save_dir = os.path.join(args.save_dir, short)

    print(f"\n{'#' * 70}")
    print(f"# MODEL: {model_name}")
    print(f"{'#' * 70}")

    try:
        # Load FP model
        fp_model = load_fp_model(model_name, device=args.device)
        tokens = tokenize_prompts(fp_model)
        n_params = sum(p.numel() for p in fp_model.parameters())
        cfg = fp_model.cfg
        print(f"Loaded: {n_params/1e6:.0f}M params, {cfg.n_layers} layers, "
              f"{cfg.n_heads} heads, d_model={cfg.d_model}")

        all_results = {}
        trained_sae = None

        # --- RTN ---
        print(f"\nCreating RTN 4-bit quantized copy...")
        rtn_model = create_quantized_model(fp_model)
        rtn_model.eval()

        # Report RTN weight MSE
        total_mse, n_w = 0.0, 0
        for (nm, p_fp), (_, p_q) in zip(fp_model.named_parameters(), rtn_model.named_parameters()):
            if "blocks" in nm and "W_" in nm:
                total_mse += (p_fp.data - p_q.data).pow(2).mean().item()
                n_w += 1
        if n_w:
            print(f"RTN weight MSE across {n_w} parameters: {total_mse / n_w:.6e}")

        rtn_save = os.path.join(model_save_dir, "rtn")
        os.makedirs(rtn_save, exist_ok=True)

        results, trained_sae = run_analyses(
            "rtn", fp_model, rtn_model, tokens, rtn_save, args, trained_sae
        )
        all_results["rtn"] = results

        del rtn_model
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

        # --- GPTQ ---
        if not args.no_gptq:
            gptq_model = create_gptq_quantized_model(
                fp_model, model_name=model_name, device=args.device
            )

            gptq_save = os.path.join(model_save_dir, "gptq")
            os.makedirs(gptq_save, exist_ok=True)

            results, trained_sae = run_analyses(
                "gptq", fp_model, gptq_model, tokens, gptq_save, args, trained_sae
            )
            all_results["gptq"] = results

            del gptq_model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

        # Comparison plot
        if len(all_results) > 1:
            plot_comparison(all_results, model_save_dir, short_name=short)

        # Print model summary
        print(f"\n{'=' * 70}")
        print(f"SUMMARY: {model_name}")
        print(f"{'=' * 70}")
        for label, res in all_results.items():
            print(f"\n--- {label.upper()} 4-bit ---")
            if "activation" in res:
                for comp, metrics in res["activation"].items():
                    print(f"  {comp} mean cosine sim: {metrics['cosine_similarity'].mean():.4f}")
            if "attention" in res:
                print(f"  Attention JSD mean: {res['attention']['jsd_matrix'].mean():.6f}")
            if "feature" in res:
                print(f"  Feature survival rate: {res['feature']['survival_rate']:.1%}")
            if "circuit" in res:
                print(f"  Induction head survival: {res['circuit']['survival_rate']:.0%}")
                print(f"  Induction score correlation: {res['circuit']['correlation']:.4f}")

        # Store for scaling study
        scaling_results[model_name] = {
            "n_params": n_params,
            "short_name": short,
            **all_results,
        }

        # Cleanup
        del fp_model, trained_sae, tokens
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        print(f"\n*** OOM on {model_name} — skipping ***")
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n*** Error on {model_name}: {e} — skipping ***")
        import traceback
        traceback.print_exc()
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Scaling study plots
# ---------------------------------------------------------------------------

def plot_scaling_study(scaling_results, save_dir):
    """Plot induction head survival rate vs model size (THE headline plot)."""
    models = sorted(scaling_results.keys(), key=lambda m: scaling_results[m]["n_params"])

    params, labels, rtn_surv, gptq_surv = [], [], [], []
    for m in models:
        r = scaling_results[m]
        rtn_circ = r.get("rtn", {}).get("circuit")
        if rtn_circ is None:
            continue
        params.append(r["n_params"])
        labels.append(r["short_name"])
        rtn_surv.append(rtn_circ["survival_rate"] * 100)
        gptq_circ = r.get("gptq", {}).get("circuit")
        gptq_surv.append(gptq_circ["survival_rate"] * 100 if gptq_circ else float("nan"))

    if len(params) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(params, rtn_surv, "ro-", linewidth=2.5, markersize=10, label="RTN 4-bit", zorder=5)
    ax.plot(params, gptq_surv, "bo-", linewidth=2.5, markersize=10, label="GPTQ 4-bit", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters", fontsize=13)
    ax.set_ylabel("Induction Head Survival Rate (%)", fontsize=13)
    ax.set_title("Scaling Study: Induction Head Survival under 4-bit Quantization",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    ax.set_xticks(params)
    tick_labels = []
    for l, p in zip(labels, params):
        pstr = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.1f}B"
        tick_labels.append(f"{l}\n({pstr})")
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)

    for x, y in zip(params, rtn_surv):
        if not np.isnan(y):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, 12), ha="center", fontsize=9, color="red")
    for x, y in zip(params, gptq_surv):
        if not np.isnan(y):
            ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                       xytext=(0, -15), ha="center", fontsize=9, color="blue")

    fig.tight_layout()
    path = os.path.join(save_dir, "scaling_study.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_scaling_table(scaling_results, save_dir):
    """Generate a summary table image with all metrics across models."""
    models = sorted(scaling_results.keys(), key=lambda m: scaling_results[m]["n_params"])

    def _get(res, *keys, fmt=".3f"):
        try:
            val = res
            for k in keys:
                val = val[k]
            if hasattr(val, "mean"):
                val = float(val.mean())
            else:
                val = float(val)
            return f"{val:{fmt}}"
        except (KeyError, TypeError, IndexError):
            return "—"

    headers = [
        "Model", "Params",
        "RTN\nResid Cos", "GPTQ\nResid Cos",
        "RTN\nAttn JSD", "GPTQ\nAttn JSD",
        "RTN\nFeat Surv", "GPTQ\nFeat Surv",
        "RTN\nInd Surv", "GPTQ\nInd Surv",
        "RTN\nInd Corr", "GPTQ\nInd Corr",
    ]

    rows = []
    for m in models:
        r = scaling_results[m]
        p = r["n_params"]
        pstr = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.1f}B"
        rtn = r.get("rtn", {})
        gptq = r.get("gptq", {})

        rows.append([
            r["short_name"], pstr,
            _get(rtn, "activation", "resid_post", "cosine_similarity"),
            _get(gptq, "activation", "resid_post", "cosine_similarity"),
            _get(rtn, "attention", "jsd_matrix", fmt=".4f"),
            _get(gptq, "attention", "jsd_matrix", fmt=".4f"),
            _get(rtn, "feature", "survival_rate", fmt=".0%"),
            _get(gptq, "feature", "survival_rate", fmt=".0%"),
            _get(rtn, "circuit", "survival_rate", fmt=".0%"),
            _get(gptq, "circuit", "survival_rate", fmt=".0%"),
            _get(rtn, "circuit", "correlation"),
            _get(gptq, "circuit", "correlation"),
        ])

    fig, ax = plt.subplots(figsize=(20, 2 + len(rows) * 0.8))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Scaling Study: All Metrics", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    path = os.path.join(save_dir, "scaling_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def print_scaling_summary(scaling_results):
    """Print the cross-model scaling study table to console."""
    models = sorted(scaling_results.keys(), key=lambda m: scaling_results[m]["n_params"])

    print(f"\n{'=' * 95}")
    print("SCALING STUDY: Induction Head Survival under 4-bit Quantization")
    print(f"{'=' * 95}")
    print(f"{'Model':<16} | {'Params':>6} | {'RTN survival':>14} | {'GPTQ survival':>14} | {'RTN corr':>9} | {'GPTQ corr':>9}")
    print("-" * 95)

    for m in models:
        r = scaling_results[m]
        p = r["n_params"]
        pstr = f"{p/1e6:.0f}M" if p < 1e9 else f"{p/1e9:.1f}B"

        rtn_circ = r.get("rtn", {}).get("circuit", {})
        gptq_circ = r.get("gptq", {}).get("circuit", {})

        n_total_rtn = len(rtn_circ.get("induction_heads", []))
        n_total_gptq = len(gptq_circ.get("induction_heads", []))
        n_total = max(n_total_rtn, n_total_gptq, 1)

        if rtn_circ:
            n_surv = int(round(rtn_circ["survival_rate"] * n_total))
            rtn_str = f"{n_surv}/{n_total} ({rtn_circ['survival_rate']:.0%})"
            rtn_corr = f"{rtn_circ['correlation']:.3f}"
        else:
            rtn_str = "—"
            rtn_corr = "—"

        if gptq_circ:
            n_surv = int(round(gptq_circ["survival_rate"] * n_total))
            gptq_str = f"{n_surv}/{n_total} ({gptq_circ['survival_rate']:.0%})"
            gptq_corr = f"{gptq_circ['correlation']:.3f}"
        else:
            gptq_str = "—"
            gptq_corr = "—"

        print(f"{r['short_name']:<16} | {pstr:>6} | {rtn_str:>14} | {gptq_str:>14} | {rtn_corr:>9} | {gptq_corr:>9}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="P2: Mechanistic Interpretability of Quantization — Scaling Study"
    )
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Single model to analyze (default: gpt2)")
    parser.add_argument("--models-all", action="store_true",
                        help="Run all 5 models for scaling study")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sae-epochs", type=int, default=None,
                        help="SAE epochs (default: adaptive based on model size)")
    parser.add_argument("--no-gptq", action="store_true",
                        help="Skip GPTQ model (faster, RTN only)")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        choices=["activation", "attention", "feature", "logit", "circuit"],
                        help="Skip specific analyses")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.init(project="p2-interpretability", config=vars(args))

    if args.models_all:
        models_to_run = MODELS_ALL
    else:
        models_to_run = [args.model]

    t_start = time.time()

    print("=" * 70)
    print("P2: MECHANISTIC INTERPRETABILITY — QUANTIZATION SCALING STUDY")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Device: {args.device}")
    print("=" * 70)

    scaling_results = {}

    for model_name in models_to_run:
        run_single_model(model_name, args, scaling_results)

    # Scaling study (multi-model only)
    if len(scaling_results) > 1:
        print(f"\n{'#' * 70}")
        print("# SCALING STUDY")
        print(f"{'#' * 70}")
        plot_scaling_study(scaling_results, args.save_dir)
        plot_scaling_table(scaling_results, args.save_dir)
        print_scaling_summary(scaling_results)

    # W&B logging
    if args.wandb:
        import wandb
        summary = {}
        for model_name, r in scaling_results.items():
            short = r["short_name"]
            for label in ["rtn", "gptq"]:
                res = r.get(label, {})
                if "activation" in res:
                    for comp, metrics in res["activation"].items():
                        summary[f"{short}/{label}/activation/{comp}/mean_cosine"] = float(metrics["cosine_similarity"].mean())
                if "attention" in res:
                    summary[f"{short}/{label}/attention/mean_jsd"] = float(res["attention"]["jsd_matrix"].mean())
                if "feature" in res:
                    summary[f"{short}/{label}/feature/survival_rate"] = float(res["feature"]["survival_rate"])
                if "circuit" in res:
                    summary[f"{short}/{label}/circuit/induction_survival"] = float(res["circuit"]["survival_rate"])
                    summary[f"{short}/{label}/circuit/score_correlation"] = float(res["circuit"]["correlation"])
        wandb.log(summary)
        wandb.finish()

    total_time = time.time() - t_start
    print(f"\nTotal runtime: {total_time:.1f}s ({total_time / 60:.1f}min)")
    print("Done.")


if __name__ == "__main__":
    main()
