"""LCG's attention mask on NVV tags (OmniVoice): does it find the tag, and does it work as the guidance window?

Phrase-Localized LCG (Lee et al. 2026, arXiv 2609.01016) localizes its contrast on OmniVoice with a mask read from
the model's own attention: for every target frame, the attention from that frame to the transcript's text tokens in
layers 8 and 12 (max over heads, then over the two layers) peaks at one token; the frame is in the mask if that
token belongs to the phrase. The mask is then dilated by a max filter of radius 10 frames (their M4). For a tag,
the "phrase" is the tag's tokens.

Two systems, same seed, voice and code path as run_omnivoice.py:

  base       no guidance; records the LCG mask at every step. The audio is the judged base clip (same_audio), so
             the masks can be split by whether the judge heard the tag.
  p2_lcg     presence guidance (l1 = 2, first 8 steps) with the LCG mask, recomputed at each step from that step's
             forward pass, in place of our duration window. Everything else as p2. Writes
             <out>/omnivoice_p2_lcg/<id>.wav and timing rows (inflated by the recording), for `bench.score`.

The dual-tag union of LCG does not apply: the presence contrast's other branch has no tag.

    python -m bench.guidance.lcg_omnivoice --out out [--systems base p2_lcg]

Writes <out>/attn/omnivoice_lcg_<system>.pt (dict id -> per-step masks); `bench.guidance.lcg_report` summarizes.
"""
import argparse
import time
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb
from bench.excite.attn_omnivoice import same_audio

LAYERS = (8, 12)
RADIUS = 10  # frames; LCG's margin dilation k = 4: radius 1 + 2 + 3 + 4
SYSTEMS = {"base": (0, 0, 0), "p2_lcg": (2, 0, 0)}


class MaskProbe:
    """Attention function for the LLM. On every forward while .q is set, takes for row 0 the attention from the
    target frames .q to the text tokens .text in LAYERS (natural, without any excitation bias), max over heads and
    layers, and appends to .argmax the text token each frame attends to most (index into .text)."""

    NAME = "tagguide_lcg_probe"

    def __init__(self, llm, torch):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

        self.torch, self.repeat_kv = torch, repeat_kv
        self.inner = ALL_ATTENTION_FUNCTIONS.get_interface(llm.config._attn_implementation, eager_attention_forward)
        self.q, self.text, self.peak, self.argmax = None, None, None, []
        AttentionInterface.register(self.NAME, self)
        llm.set_attn_implementation(self.NAME)

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if self.q is not None and module.layer_idx in LAYERS:
            q = query[0:1, :, self.q, :].float()
            k = self.repeat_kv(key[0:1], module.num_key_value_groups).float()
            score = (q @ k.transpose(-1, -2)) * module.scaling  # [1, H, T, L]
            mask = attention_mask[0:1, :, self.q, :k.size(-2)]
            if mask.dtype == self.torch.bool:
                score = score.masked_fill(~mask, -float("inf"))
            else:
                score = score + mask.float().clamp(max=0)
            peak = score.softmax(-1)[0][:, :, self.text].amax(0)  # [T, text]
            self.peak = peak if module.layer_idx == LAYERS[0] else self.torch.maximum(self.peak, peak)
            if module.layer_idx == LAYERS[-1]:
                self.argmax.append(self.peak.argmax(-1))
        return self.inner(module, query, key, value, attention_mask, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--ids", nargs="*", default=None, help="items (default: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from omnivoice.models.omnivoice import _combine_text

    items = [it for it in sb.items("omnivoice") if args.ids is None or it["id"] in args.ids]
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    probe = MaskProbe(model.llm, torch)
    state = {}
    window, positions = guide.window, guide.tag_token_positions

    def traced_window(task, text):
        state["win"] = window(task, text)
        return state["win"]

    def traced_positions(task, text, cond_len, grouped=False):
        """Set the probe's queries (target frames) and keys (the transcript's text tokens) for row 0."""
        keys = positions(task, text, cond_len, grouped)
        head, tail = "<|text_start|>", "<|text_end|>"
        wrapped = f"{head}{_combine_text(ref_text=task.ref_texts[0], text=text)}{tail}"
        enc = model.text_tokenizer(wrapped, return_offsets_mapping=True)
        t_len = task.target_lens[0]
        style_len = cond_len - len(enc.input_ids) - task.ref_audio_tokens[0].size(-1) - t_len
        text_pos = [style_len + i for i, (a, b) in enumerate(enc.offset_mapping)
                    if b > a and a >= len(head) and b <= len(wrapped) - len(tail)]
        probe.q = torch.arange(cond_len - t_len, cond_len, device=model.device)
        probe.text = torch.tensor(text_pos, device=model.device)
        state["tag_rel"] = torch.tensor([text_pos.index(p) for p in keys], device=model.device)
        return keys

    def lcg_mask(argmax):
        """Raw LCG mask ([T] bool: frames whose peak token is the tag's) and its dilation."""
        raw = torch.isin(argmax, state["tag_rel"])
        dil = F.max_pool1d(raw.float().view(1, 1, -1), 2 * RADIUS + 1, 1, RADIUS).view(-1) > 0
        return raw, dil

    def lcg_window(step, w):
        _, dil = lcg_mask(probe.argmax[-1])
        return dil.view(1, 1, -1, 1).float()

    guide.window, guide.tag_token_positions = traced_window, traced_positions
    (args.out / "attn").mkdir(exist_ok=True)
    done = sb.done_rows(args.out)
    for name in args.systems:
        guide.setting = SYSTEMS[name]
        guide.step_window = lcg_window if name == "p2_lcg" else type(guide).step_window.__get__(guide)
        system = f"omnivoice_{name}"
        (args.out / system).mkdir(exist_ok=True)
        result = {}
        for k, it in enumerate(items):
            probe.q, probe.argmax = None, []
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - t0
            probe.q = None
            raw = torch.stack([lcg_mask(a)[0] for a in probe.argmax])  # [steps, T]
            wav = args.out / system / f"{it['id']}.wav"
            if name == "base":
                same = same_audio(audio, wav, torch, sf)
            else:
                same = None
                if (system, it["id"]) not in done:
                    audio = torch.as_tensor(audio).squeeze().float().cpu().numpy()
                    sf.write(wav, audio, model.sampling_rate)
                    sb.append_jsonl(args.out / "timing.jsonl",
                                    {"id": it["id"], "system": system, "type": it["type"], "gen_s": round(gen_s, 3),
                                     "out_s": round(len(audio) / model.sampling_rate, 2)})
            result[it["id"]] = {"raw": raw.cpu(), "argmax": torch.stack(probe.argmax).short().cpu(),
                                "tag_rel": state["tag_rel"].cpu(), "window": state["win"].cpu(),
                                "type": it["type"], "same_audio": same}
            print(f"[{name} {k + 1}/{len(items)}] {it['id']} raw frames step8={int(raw[min(7, len(raw) - 1)].sum())}"
                  f" last={int(raw[-1].sum())} same_audio={same}", flush=True)
        torch.save(result, args.out / "attn" / f"omnivoice_lcg_{name}.pt")


if __name__ == "__main__":
    main()
