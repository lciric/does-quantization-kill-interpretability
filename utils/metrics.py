"""
Metrics for comparing full-precision and quantized model representations.

Provides cosine similarity, L2 distance, Pearson correlation, KL divergence,
Jensen-Shannon divergence, and feature survival rate computations.
"""

import torch
import torch.nn.functional as F


def cosine_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between corresponding vectors in two batches.

    Args:
        a, b: Tensors of shape (..., d). Compared along last dimension.

    Returns:
        Cosine similarities, shape (...).
    """
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos


def l2_distance_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    L2 distance between corresponding vectors.

    Args:
        a, b: Tensors of shape (..., d).

    Returns:
        L2 distances, shape (...).
    """
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    return torch.norm(a_flat - b_flat, dim=-1)


def pearson_correlation_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation between corresponding vectors.

    Args:
        a, b: Tensors of shape (..., d).

    Returns:
        Correlations, shape (...).
    """
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()

    a_centered = a_flat - a_flat.mean(dim=-1, keepdim=True)
    b_centered = b_flat - b_flat.mean(dim=-1, keepdim=True)

    num = (a_centered * b_centered).sum(dim=-1)
    denom = a_centered.norm(dim=-1) * b_centered.norm(dim=-1)

    return num / denom.clamp(min=1e-8)


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    KL divergence KL(P || Q) from logits.

    Args:
        p_logits, q_logits: Tensors of shape (..., vocab_size).

    Returns:
        KL divergence per position, shape (...).
    """
    p = F.softmax(p_logits.float(), dim=-1)
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor, from_logits: bool = False
) -> torch.Tensor:
    """
    Jensen-Shannon divergence between two distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P+Q).

    Args:
        p, q: Probability distributions (or logits if from_logits=True).
               Shape (..., n_classes).
        from_logits: If True, apply softmax first.

    Returns:
        JSD values, shape (...).
    """
    if from_logits:
        p = F.softmax(p.float(), dim=-1)
        q = F.softmax(q.float(), dim=-1)
    else:
        p = p.float()
        q = q.float()

    m = 0.5 * (p + q)

    # Use xlogy(a, b) = a * log(b), with xlogy(0, b) = 0 (no NaN)
    kl_pm = (torch.xlogy(p, p) - torch.xlogy(p, m)).sum(dim=-1)
    kl_qm = (torch.xlogy(q, q) - torch.xlogy(q, m)).sum(dim=-1)

    return 0.5 * kl_pm + 0.5 * kl_qm


def feature_survival_rate(
    top_tokens_fp: list[list[int]],
    top_tokens_q: list[list[int]],
    threshold: float = 0.8,
) -> tuple[float, list[float]]:
    """
    Compute feature survival rate: fraction of features whose top-k activating
    tokens remain mostly identical between FP and quantized models.

    A feature "survives" if the overlap between its top-k tokens in FP and
    quantized models exceeds the threshold.

    Args:
        top_tokens_fp: List of lists, each containing top-k token indices for a feature (FP).
        top_tokens_q: Same for quantized model.
        threshold: Minimum overlap fraction for a feature to count as "survived".

    Returns:
        (survival_rate, per_feature_overlap): Overall rate and per-feature overlap scores.
    """
    assert len(top_tokens_fp) == len(top_tokens_q)

    overlaps = []
    survived = 0

    for fp_tokens, q_tokens in zip(top_tokens_fp, top_tokens_q):
        fp_set = set(fp_tokens)
        q_set = set(q_tokens)
        if len(fp_set) == 0:
            overlaps.append(0.0)
            continue
        overlap = len(fp_set & q_set) / len(fp_set)
        overlaps.append(overlap)
        if overlap >= threshold:
            survived += 1

    rate = survived / len(top_tokens_fp) if len(top_tokens_fp) > 0 else 0.0
    return rate, overlaps


def top_k_agreement(
    logits_fp: torch.Tensor,
    logits_q: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Fraction of top-k predicted tokens that agree between FP and quantized logits.

    Args:
        logits_fp, logits_q: Shape (..., vocab_size).
        k: Number of top predictions to compare.

    Returns:
        Agreement fractions, shape (...).
    """
    top_fp = logits_fp.float().topk(k, dim=-1).indices  # (..., k)
    top_q = logits_q.float().topk(k, dim=-1).indices

    # Count how many of q's top-k appear in fp's top-k
    # Expand for broadcasting comparison
    fp_expanded = top_fp.unsqueeze(-1)  # (..., k, 1)
    q_expanded = top_q.unsqueeze(-2)    # (..., 1, k)
    matches = (fp_expanded == q_expanded).any(dim=-1).float().sum(dim=-1)  # (...,)

    return matches / k
