"""Tag-local guidance for CosyVoice 2 (autoregressive Qwen2 LM over speech tokens, then flow matching).

CosyVoice 2's CFG lives in the flow matcher, which sees speech tokens, not the text, so the LM has no
unconditional pass (g = 0). The terms act on the LM's log-probabilities:

    presence  + l1 * m * (c - c_notag)    a second LM stream on the text with the tag removed
    type      + l2 * m * (c - c_conf)     a third stream with the tag replaced by its confuser
    excite    + b on the attention score from the current speech token to the tag's text tokens, while m = 1

All streams are fed the same sampled speech tokens. The output length is open, so the window m comes from the
model: the cond stream's attention over the target text (mean over heads and the alignment layers, without the
excitation bias) peaks at a token, and m = 1 while that token lies within the word before the tag to the word after
it, widened by `dilate` tokens. The terms act at every step in the window. Sampling is CosyVoice's own.

    reader = Reader(cosy.model.llm.llm.model)
    job = make_job(hf_tokenizer, encode, text, tag, notag, confuser, setting=(2, 3, 2))
    cosy.model.llm.inference = guided_inference(cosy.model.llm, reader, job)
    cosy.inference_zero_shot(text, ref_text, ref_wav, stream=False, text_frontend=False)
"""
import types

import torch
import torch.nn.functional as F

from .tags import word_span


class Reader:
    """Attention function for the LM's layers. On single-token steps while .on is set it records the attention of
    the query over the target text keys .keys (mean over heads, before any bias) in .mass, one row per layer, and
    adds .bias to the scores on the keys .tag when .excite is set."""

    NAME = "tagguide_reader"

    def __init__(self, qwen):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.models.qwen2.modeling_qwen2 import repeat_kv

        self.repeat_kv = repeat_kv
        self.inner = ALL_ATTENTION_FUNCTIONS[qwen.config._attn_implementation]
        self.on, self.excite, self.keys, self.tag, self.bias, self.mass = False, False, None, None, 0.0, []
        AttentionInterface.register(self.NAME, self)
        qwen.config._attn_implementation = self.NAME

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if not self.on or query.shape[2] != 1:
            return self.inner(module, query, key, value, aligned(attention_mask, key.shape[2], query.dtype), **kwargs)
        k = self.repeat_kv(key, module.num_key_value_groups).float()
        s = (query.float() @ k.transpose(-1, -2))[0, :, 0] * module.scaling  # [H, L]
        self.mass.append(s[:, self.keys].softmax(-1).mean(0))
        if self.excite:
            add = torch.zeros((1, 1, 1, key.shape[2]), dtype=query.dtype, device=query.device)
            add[..., self.tag] = self.bias
            if attention_mask is not None:
                am = attention_mask[..., :key.shape[2]]
                add = add + (am.to(query.dtype) if am.dtype != torch.bool
                             else torch.zeros_like(am, dtype=query.dtype).masked_fill(~am, torch.finfo(query.dtype).min))
            attention_mask = add
        return self.inner(module, query, key, value, aligned(attention_mask, key.shape[2], query.dtype), **kwargs)


def aligned(mask, n_keys, dtype):
    """A custom attention name makes Qwen2 pass a 4D mask, which SDPA's memory-efficient kernel takes only with a
    last-dimension stride that is a multiple of 8: copy it, as float, into an 8-aligned buffer."""
    if mask is None:
        return None
    mask = mask[..., :n_keys]
    if mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, torch.finfo(dtype).min)
    buf = torch.zeros((*mask.shape[:-1], -(-n_keys // 8) * 8), dtype=mask.dtype, device=mask.device)
    buf[..., :n_keys] = mask
    return buf[..., :n_keys]


def make_job(hf_tok, encode, text, tag, notag, confuser, setting, dilate=1, layers=(11, 16)):
    """Inputs of guided_inference for one utterance. hf_tok: CosyVoice's HF tokenizer (offsets); encode(text):
    CosyVoice's text frontend, returning a [1, N] tensor of text tokens. Positions are in the target text tokens."""
    enc = hf_tok(text, return_offsets_mapping=True)
    assert enc.input_ids == encode(text)[0].tolist(), "HF tokenizer and CosyVoice frontend disagree"
    s = text.find(tag)
    a, b = word_span(text, s, s + len(tag))

    def span(lo, hi):
        return [i for i, (x, y) in enumerate(enc.offset_mapping) if y > x and y > lo and x < hi]

    win = span(a, b)
    return {"setting": tuple(setting), "notag": encode(notag), "confuser": encode(confuser),
            "tag": span(s, s + len(tag)), "window": (win[0], win[-1] + 1), "dilate": dilate, "layers": list(layers)}


def read_position(mass, layers):
    """Read position from one step's attention rows ([layers, N]): peak of the alignment layers' mean, smoothed."""
    m = mass[layers].mean(0)
    return int(F.avg_pool1d(m.view(1, 1, -1), 3, 1, 1).argmax())


def guided_inference(lm, reader, job):
    """A replacement for Qwen2LM.inference, bound to `lm`, for one utterance described by `job` (make_job). After
    the run, job["in_window"] is the share of steps the read position spent at the tag."""

    @torch.inference_mode()
    def inference(self, text, text_len, prompt_text, prompt_text_len, prompt_speech_token, prompt_speech_token_len,
                  embedding, sampling=25, max_token_text_ratio=20, min_token_text_ratio=2, uuid=""):
        l1, l2, bias = job["setting"]
        dev = text.device
        sos = self.llm_embedding.weight[self.sos].reshape(1, 1, -1)
        task = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        speech = self.speech_embedding(prompt_speech_token) if prompt_speech_token_len != 0 else None

        def lm_input(tokens):
            emb = self.llm.model.model.embed_tokens(torch.cat([prompt_text, tokens.to(dev)], dim=1))
            return torch.cat([sos, emb, task] + ([speech] if speech is not None else []), dim=1)

        streams = [lm_input(text)]
        if l1:
            streams.append(lm_input(job["notag"]))
        if l2:
            streams.append(lm_input(job["confuser"]))
        n_text = text.shape[1]
        min_len, max_len = int(n_text * min_token_text_ratio), int(n_text * max_token_text_ratio)
        start = 1 + prompt_text.shape[1]  # first target text position in the cond stream
        reader.keys = torch.arange(start, start + n_text, device=dev)
        reader.tag = torch.tensor(job["tag"], device=dev) + start
        reader.bias = bias
        lo, hi = job["window"][0] - job["dilate"], job["window"][1] + job["dilate"]

        caches, out_tokens, in_win, inside = [None] * len(streams), [], False, 0
        inputs = streams
        for i in range(max_len):
            lps = []
            for r, x in enumerate(inputs):
                reader.on = r == 0
                reader.excite = r == 0 and bool(bias) and in_win
                reader.mass = []
                masks = torch.tril(torch.ones((1, x.shape[1], x.shape[1]), device=dev)).to(torch.bool)
                y, caches[r] = self.llm.forward_one_step(x, masks=masks, cache=caches[r])
                lps.append(self.llm_decoder(y[:, -1]).log_softmax(dim=-1))
                if r == 0:  # the prefill step (i = 0) has no single-query attention
                    pos = -1 if i == 0 else read_position(torch.stack(reader.mass), job["layers"])
            reader.on = reader.excite = False
            c = lps[0]
            if in_win:
                j = 1
                if l1:
                    c = c + l1 * (lps[0] - lps[j])
                    j += 1
                if l2:
                    c = c + l2 * (lps[0] - lps[j])
                c = c.log_softmax(dim=-1)
            top = self.sampling_ids(c.squeeze(0), out_tokens, sampling, ignore_eos=i < min_len)
            at_tag = lo <= pos < hi
            inside += at_tag
            in_win = bool(l1 or l2 or bias) and at_tag
            if top in self.stop_token_ids:
                break
            yield top
            out_tokens.append(top)
            nxt = self.speech_embedding.weight[top].reshape(1, 1, -1)
            inputs = [nxt] * len(streams)
        job["in_window"] = inside / max(len(out_tokens), 1)

    return types.MethodType(inference, lm)
