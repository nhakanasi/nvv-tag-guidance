"""Attention excitation: a constant bias b on the attention scores from the output in the tag's window to the tag's
text tokens, in every layer and head of the conditional pass,

    s'_ij = s_ij + b [i in window, j in tag],      p'_iT = p_iT e^b / (1 - p_iT + p_iT e^b),

so the odds of attending to the tag are multiplied by e^b. Unlike Attend-and-Excite (a gradient step on the latent),
it needs no gradient and no extra pass; the window supplies the location the boost acts from. See
bench/excite/README.md for the motivation and the evidence.
"""
import torch


def odds_boost(p, b):
    """Attention share on the tag after the bias, from the share p before it."""
    e = torch.exp(torch.as_tensor(b, dtype=torch.float32))
    return p * e / (1 - p + p * e)


def additive_mask(allow, row, queries, keys, bias, dtype):
    """Float attention mask from a boolean one (`allow`, [B, 1, L, L]) with +bias from the query positions to the key
    positions in batch row `row` (OmniVoice: the conditional row, queries = the window's frames)."""
    mask = torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill(~allow, torch.finfo(dtype).min)
    mask[row, 0, queries.unsqueeze(-1), keys.unsqueeze(0)] += bias
    return mask


def bias_row(attention_mask, n_keys, keys, bias, dtype, device):
    """Additive mask for a single query step ([1, 1, 1, n_keys]): +bias on `keys`, merged with the model's own mask
    (CosyVoice 2: the current speech token, keys = the tag's text tokens)."""
    add = torch.zeros((1, 1, 1, n_keys), dtype=dtype, device=device)
    add[..., keys] = bias
    if attention_mask is not None:
        am = attention_mask[..., :n_keys]
        add = add + (am.to(dtype) if am.dtype != torch.bool
                     else torch.zeros_like(am, dtype=dtype).masked_fill(~am, torch.finfo(dtype).min))
    return add
