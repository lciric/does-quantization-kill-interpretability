"""
Self-contained GPTQ quantization for transformer models.

Implements the GPTQ algorithm (Frantar et al., 2022) with support for:
  - GPT-2 family (gpt2, gpt2-xl) — Conv1D layers
  - Pythia / GPT-NeoX family (EleutherAI/pythia-*) — Linear layers

Uses WikiText-2 calibration data, processes layers block-by-block
with hook-based Hessian accumulation.
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_row_scales(W, n_bits=4):
    """Per-row quantization scales for weight matrix (out_features, in_features)."""
    qmax = 2 ** (n_bits - 1) - 1
    scales = W.abs().amax(dim=1) / qmax
    return scales.clamp(min=1e-10)


def quantize_column(w_col, scales, n_bits=4):
    """Quantize a single column using pre-computed per-row scales."""
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    q = torch.clamp(torch.round(w_col / scales), qmin, qmax)
    return q * scales


def gptq_quantize_layer(W, H, n_bits=4, block_size=128):
    """
    GPTQ quantization of a single weight matrix (Algorithm 1).

    Args:
        W: Weight matrix, shape (out_features, in_features). Float32.
        H: Hessian, shape (in_features, in_features). Float32.
        n_bits: Quantization bit-width.
        block_size: Columns per block.

    Returns:
        Q: Quantized weights (dequantized to float), same shape as W.
        loss: Scalar quantization loss.
    """
    W = W.clone().float()
    n_rows, n_cols = W.shape

    # Invert H via Cholesky
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        extra_damp = 0.1 * torch.mean(torch.diag(H))
        H_reg = H.clone()
        diag_idx = range(H.shape[0])
        H_reg[diag_idx, diag_idx] += extra_damp
        L = torch.linalg.cholesky(H_reg)
        H_inv = torch.cholesky_inverse(L)

    scales = compute_row_scales(W, n_bits)
    Q = torch.zeros_like(W)
    loss = 0.0

    for i1 in range(0, n_cols, block_size):
        i2 = min(i1 + block_size, n_cols)
        count = i2 - i1

        W_blk = W[:, i1:i2].clone()
        Q_blk = torch.zeros_like(W_blk)
        Err = torch.zeros_like(W_blk)
        H_inv_blk = H_inv[i1:i2, i1:i2]

        for j in range(count):
            w_col = W_blk[:, j]
            h_jj = H_inv_blk[j, j]

            q_col = quantize_column(w_col, scales, n_bits)
            Q_blk[:, j] = q_col

            err = (w_col - q_col) / h_jj
            Err[:, j] = err

            loss += ((w_col - q_col) ** 2 / h_jj).sum().item()

            if j < count - 1:
                W_blk[:, j + 1 :] -= (
                    err.unsqueeze(1) * H_inv_blk[j, j + 1 :].unsqueeze(0)
                )

        Q[:, i1:i2] = Q_blk
        if i2 < n_cols:
            W[:, i2:] -= Err @ H_inv[i1:i2, i2:]

    return Q, loss


def get_calibration_data(tokenizer, n_samples=128, seq_len=1024, seed=42):
    """
    Extract calibration data from WikiText-2 training set.

    Returns a list of tensors, each shape (1, seq_len).
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)

    rng = torch.Generator()
    rng.manual_seed(seed)

    samples = []
    for _ in range(n_samples):
        start = torch.randint(0, len(tokens) - seq_len, (1,), generator=rng).item()
        segment = tokens[start : start + seq_len].unsqueeze(0)
        samples.append(segment)

    return samples


def _get_blocks_and_embed(model):
    """Detect model architecture and return (blocks, embed_fn)."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 family
        blocks = model.transformer.h

        def embed_fn(input_ids, device):
            pos_ids = torch.arange(0, input_ids.shape[1], device=device).unsqueeze(0)
            return model.transformer.wte(input_ids) + model.transformer.wpe(pos_ids)

        return blocks, embed_fn

    elif hasattr(model, "gpt_neox"):
        # Pythia / GPT-NeoX family
        blocks = model.gpt_neox.layers

        def embed_fn(input_ids, device):
            return model.gpt_neox.embed_in(input_ids)

        return blocks, embed_fn

    else:
        raise ValueError(f"Unsupported architecture: {type(model).__name__}")


@torch.no_grad()
def gptq_quantize_model(
    model_name="gpt2", device="cuda", n_bits=4, block_size=128, n_samples=128, seq_len=1024
):
    """
    Load a model, apply GPTQ block-by-block, return the quantized HF model.

    Supports GPT-2 family (Conv1D) and Pythia/GPT-NeoX family (Linear).

    Args:
        model_name: HuggingFace model name (e.g. "gpt2", "EleutherAI/pythia-410m").
        device: "cuda" or "cpu".
        n_bits: Quantization bit-width.
        block_size: GPTQ block size.
        n_samples: Number of calibration samples from WikiText-2.
        seq_len: Sequence length for calibration.

    Returns:
        model: Quantized HuggingFace model.
    """
    print(f"Loading {model_name} (HuggingFace) for GPTQ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    print(f"Loading WikiText-2 calibration data ({n_samples} samples, seq_len={seq_len})...")
    calibration_data = get_calibration_data(
        tokenizer, n_samples=n_samples, seq_len=seq_len
    )

    blocks, embed_fn = _get_blocks_and_embed(model)

    # Pre-compute rotary position embeddings for GPT-NeoX / Pythia
    # (transformers >= 4.44 requires explicit position_embeddings=(cos, sin))
    rotary_emb = None
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "rotary_emb"):
        rotary_emb = model.gpt_neox.rotary_emb

    def _block_forward(block, h):
        """Call block.forward with rotary embeddings if needed."""
        if rotary_emb is not None:
            pos_ids = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
            cos, sin = rotary_emb(h, pos_ids)
            return block(h, position_embeddings=(cos, sin))
        return block(h)

    # Compute embeddings -> hidden states entering block 0
    print("Computing embedding outputs...")
    hidden_states = []
    for sample in calibration_data:
        sample = sample.to(device)
        h = embed_fn(sample, device)
        hidden_states.append(h.cpu())

    if device == "cuda":
        torch.cuda.empty_cache()

    total_loss = 0.0
    n_quantized = 0

    for block_idx, block in enumerate(tqdm(blocks, desc=f"GPTQ {n_bits}-bit")):
        # Find all Conv1D / Linear layers in this block
        layers_to_quantize = []
        for name, module in block.named_modules():
            if type(module).__name__ == "Conv1D" or isinstance(module, torch.nn.Linear):
                layers_to_quantize.append((name, module))

        # Accumulate Hessians via forward hooks
        layer_H = {name: None for name, _ in layers_to_quantize}
        layer_n = {name: 0 for name, _ in layers_to_quantize}

        hooks = []
        for name, module in layers_to_quantize:

            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    H = x.T @ x
                    if layer_H[layer_name] is None:
                        layer_H[layer_name] = H
                    else:
                        layer_H[layer_name] += H
                    layer_n[layer_name] += x.shape[0]

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(name)))

        # Run calibration through this block to accumulate Hessians
        for h in hidden_states:
            _block_forward(block, h.to(device))

        for handle in hooks:
            handle.remove()

        # Quantize each layer using accumulated Hessian
        for name, module in layers_to_quantize:
            H = layer_H[name] / layer_n[name]
            damp = 0.01 * torch.mean(torch.diag(H))
            H[range(H.shape[0]), range(H.shape[0])] += damp

            # Get weight in (out, in) format — Conv1D stores (in, out)
            is_conv1d = type(module).__name__ == "Conv1D"
            if is_conv1d:
                W = module.weight.data.T.clone().float()
            else:
                W = module.weight.data.clone().float()

            Q, loss = gptq_quantize_layer(W, H, n_bits=n_bits, block_size=block_size)

            # Write back
            if is_conv1d:
                module.weight.data = Q.T.to(module.weight.dtype)
            else:
                module.weight.data = Q.to(module.weight.dtype)

            total_loss += loss
            n_quantized += 1
            del H, W, Q
            layer_H[name] = None

        del layer_H, layer_n
        if device == "cuda":
            torch.cuda.empty_cache()

        # Propagate hidden states through the (now quantized) block
        new_hidden = []
        for h in hidden_states:
            out = _block_forward(block, h.to(device))
            new_hidden.append(out[0].detach().cpu())
        hidden_states = new_hidden

    print(f"GPTQ complete: {n_quantized} layers quantized, total loss = {total_loss:.2f}")
    return model
