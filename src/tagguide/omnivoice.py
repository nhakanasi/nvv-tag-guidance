"""OmniVoice (masked diffusion over audio tokens, fixed length) with the two tag-local methods.

Each decoding step scores the masked frames with the conditional pass (voice prompt + text) and the unconditional
one (the target frames alone) and samples from  c + g (c - u),  OmniVoice's own CFG. `Guide` adds, on a window of
frames around each tag and in the first K steps only:

    guidance (contrast.py)    + l1 * m * (c - c_notag) + l2 * m * (c - c_confuser)
    excitation (excite.py)    + b on the attention scores from the window's frames to the tag's text tokens, in every
                              layer and head of the conditional pass

The contrasts are in log-probabilities. m is 1 on the frames from the start of the word before the tag to the end of
the word after it, dilated by `dilate_s` seconds, placed with OmniVoice's own duration estimate of the text up to
each point, scaled to the target length. The first K steps run one pass per stream (cond, notag, confuser, empty);
later steps run cond and empty only.

    model = OmniVoice.from_pretrained(...)
    patch_tokenization()
    guide = Guide(model, TagSet({"laughter": "sigh", ...}), steps=8, dilate_s=0.3)
    guide.setting = (2, 3, 2)          # (l1, l2, b); None: OmniVoice's own loop
    audio = model.generate(text="I can't believe it [laughter] you came.", ...)
"""
import math

import torch
import torch.nn.functional as F
from omnivoice.models import omnivoice as ov

from . import contrast, excite
from .tags import word_span


def patch_tokenization():
    """Tokenize the whole text at once, as OmniVoice's training does. Upstream generate() tokenizes each tag apart
    from the text around it (" [laughter]" becomes " ", "[", ... instead of " [", ...), so the tag tokens differ
    between training and inference. Every setting, including the base model, runs with this patch."""
    ov._tokenize_with_nonverbal_tags = lambda text, tokenizer: tokenizer(text, return_tensors="pt").input_ids


class Guide:
    """Replaces OmniVoice._generate_iterative for single-item calls whose text has a tag, while .setting is set.
    Setting (0, 0, 0) runs Guide's own copy of OmniVoice's decoding loop with no extra term (the benchmark's base,
    so base and guided clips share one code path)."""

    def __init__(self, model, tags, steps=8, dilate_s=0.3):
        self.model, self.tags, self.steps, self.dilate_s = model, tags, steps, dilate_s
        self.setting = None  # (l1 presence, l2 type, b excitation)
        # Steps [start, end) the excitation acts in; None: the same first K steps as the contrasts. Other ranges
        # are for analysis (bench/control_omnivoice.py), e.g. only the late steps.
        self.excite_steps = None
        # Steps [start, end) the contrasts act in (all streams run); None: the first K. Other ranges are for
        # analysis (bench/guidance/control_omnivoice.py).
        self.contrast_steps = None
        self.orig = model._generate_iterative
        model._generate_iterative = self

    def __call__(self, task, gen_config):
        if not self.setting or task.batch_size != 1 or not self.tags.has_tag(task.texts[0]):
            return self.orig(task, gen_config)
        return [self.run(task, gen_config)]

    def inputs(self, task, text, gen_config):
        return self.model._prepare_inference_inputs(text, task.target_lens[0], task.ref_texts[0],
                                                    task.ref_audio_tokens[0], task.langs[0], task.instructs[0],
                                                    gen_config.denoise)

    def window(self, task, text):
        """[T] bool mask over the target frames: the word before each tag to the word after it, dilated."""
        m, t_len = self.model, task.target_lens[0]
        ref_text, n_ref = task.ref_texts[0], task.ref_audio_tokens[0].size(-1)
        plain = self.tags.remove(text)
        total = m.duration_estimator.estimate_duration(plain, ref_text, n_ref)

        def frames(prefix):
            return m.duration_estimator.estimate_duration(prefix, ref_text, n_ref) * t_len / total if prefix else 0.0

        dil = self.dilate_s * m.audio_tokenizer.config.frame_rate
        mask = torch.zeros(t_len, dtype=torch.bool, device=m.device)
        removed = 0
        for match in self.tags.with_space.finditer(text):
            p = match.start() - removed  # the tag's position in the plain text
            removed += match.end() - match.start()
            start, end = word_span(plain, p, p)
            a = max(0, int(frames(plain[:start]) - dil))
            b = min(t_len, int(frames(plain[:end]) + dil) + 1)
            mask[a:b] = True
        return mask

    def step_window(self, step, w):
        """The window the contrasts use at `step` ([1, 1, T, 1] float), called after that step's forward pass.
        Override to localize differently (bench/guidance/lcg_omnivoice.py does, with LCG's attention mask)."""
        return w

    def tag_token_positions(self, task, text, cond_len, grouped=False):
        """Positions of the tags' text tokens in the conditional input (one list per tag if grouped). Override to
        excite other tokens (bench/control_omnivoice.py does, for its placebo)."""
        wrapped = f"<|text_start|>{ov._combine_text(ref_text=task.ref_texts[0], text=text)}<|text_end|>"
        enc = self.model.text_tokenizer(wrapped, return_offsets_mapping=True)
        spans = [(mt.start(), mt.end()) for mt in self.tags.span.finditer(wrapped)]
        spans = spans[-self.tags.count(text):]  # the target's tags, not any in the reference text
        style_len = cond_len - len(enc.input_ids) - task.ref_audio_tokens[0].size(-1) - task.target_lens[0]
        groups = [[style_len + i for i, (a, b) in enumerate(enc.offset_mapping) if b > a and a < e and b > s]
                  for s, e in spans]
        return groups if grouped else [i for g in groups for i in g]

    def run(self, task, gen_config):
        m, bias = self.model, self.setting[2]
        text, t_len = task.texts[0], task.target_lens[0]
        conds = [self.inputs(task, t, gen_config)
                 for t in contrast.texts(self.setting, text, self.tags.remove(text), self.tags.swap(text))]
        win = self.window(task, text)

        # One batch: the cond streams, then the unconditional row (target frames only), padded to one length.
        c_lens = [c["input_ids"].size(2) for c in conds]
        L, C, n = max(c_lens), m.config.num_audio_codebook, len(conds) + 1
        dtype = next(m.parameters()).dtype
        ids = torch.full((n, C, L), m.config.audio_mask_id, dtype=torch.long, device=m.device)
        amask = torch.zeros((n, L), dtype=torch.bool, device=m.device)
        allow = torch.zeros((n, 1, L, L), dtype=torch.bool, device=m.device)
        for i, c in enumerate(conds):
            ids[i, :, :c_lens[i]] = c["input_ids"]
            amask[i, :c_lens[i]] = c["audio_mask"]
            allow[i, :, :c_lens[i], :c_lens[i]] = True
        ids[-1, :, :t_len] = conds[0]["input_ids"][..., -t_len:]
        amask[-1, :t_len] = conds[0]["audio_mask"][..., -t_len:]
        allow[-1, :, :t_len, :t_len] = True
        for i, used in enumerate(c_lens + [t_len]):  # padding attends to itself only
            pad = torch.arange(used, L, device=m.device)
            allow[i, :, pad, pad] = True

        keys = torch.tensor(self.tag_token_positions(task, text, c_lens[0]), device=m.device)
        excited = None
        if bias:  # +b from the window's frames to the tag's tokens, conditional row only
            excited = excite.additive_mask(allow, 0, c_lens[0] - t_len + torch.nonzero(win).squeeze(-1), keys, bias,
                                           dtype)
        late, late_L = [0, n - 1], max(c_lens[0], t_len)

        # OmniVoice's own unmasking schedule.
        tokens = torch.full((1, C, t_len), m.config.audio_mask_id, dtype=torch.long, device=m.device)
        ts = ov._get_time_steps(0.0, 1.0, gen_config.num_step, gen_config.t_shift).tolist()
        total, rem, sched = t_len * C, t_len * C, []
        for s in range(gen_config.num_step):
            k = rem if s == gen_config.num_step - 1 else min(math.ceil(total * (ts[s + 1] - ts[s])), rem)
            sched.append(int(k))
            rem -= int(k)
        layer_ids = torch.arange(C, device=m.device).view(1, -1, 1)
        w = win.view(1, 1, -1, 1).float()

        e_lo, e_hi = self.excite_steps or (0, self.steps)
        c_lo, c_hi = self.contrast_steps or (0, self.steps)
        for s in range(gen_config.num_step):
            early = c_lo <= s < c_hi
            mask = excited if excited is not None and e_lo <= s < e_hi else allow
            if early:
                logits = m(input_ids=ids, audio_mask=amask, attention_mask=mask).logits.float()
                n_rows = n
            else:
                logits = m(input_ids=ids[late, :, :late_L], audio_mask=amask[late, :late_L],
                           attention_mask=mask[late][:, :, :late_L, :late_L]).logits.float()
                n_rows = 2
            if sched[s] <= 0:
                continue

            def stream(r):
                return F.log_softmax(logits[r:r + 1, :, c_lens[r] - t_len:c_lens[r], :], dim=-1)

            c = stream(0)
            u = F.log_softmax(logits[n_rows - 1:n_rows, :, :t_len, :], dim=-1)
            lp = c + gen_config.guidance_scale * (c - u)
            if early:
                lp = contrast.apply(lp, c, [stream(j) for j in range(1, len(conds))], self.setting,
                                    self.step_window(s, w))
            lp = torch.log_softmax(lp, dim=-1)
            lp[..., m.config.audio_mask_id] = -float("inf")
            pred = lp.argmax(dim=-1)
            scores = lp.max(dim=-1)[0] - layer_ids * gen_config.layer_penalty_factor
            if gen_config.position_temperature > 0.0:
                scores = ov._gumbel_sample(scores, gen_config.position_temperature)
            scores.masked_fill_(tokens != m.config.audio_mask_id, -float("inf"))
            _, idx = torch.topk(scores.flatten(), sched[s])
            flat = tokens.flatten()
            flat[idx] = pred.flatten()[idx]
            tokens = flat.view_as(tokens)
            for i in range(n - 1):
                ids[i, :, c_lens[i] - t_len:c_lens[i]] = tokens[0]
            ids[-1, :, :t_len] = tokens[0]
        return tokens[0]
