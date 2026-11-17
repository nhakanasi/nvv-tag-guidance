"""Attention trace of excitation on OmniVoice: how much the frames around a tag attend to the tag's tokens.

Re-synthesizes items in the given settings (same seed, voice and code path as run_omnivoice.py) and records, at
every decoding step and in every layer, the conditional pass's attention mass (mean over heads) from each target
frame to the tag's text tokens:

  natural   the model's own attention, with the excitation bias taken out of the mask
  excited   the attention actually used (equal to natural when b = 0)

Per item it saves, as [step, layer] arrays, the mean over the tag window's frames (in_*) and over all other target
frames (out_*), the per-frame natural mass (mean over layers), the window and the step count K excitation acts in.
The re-synthesized audio is compared with the clip in --out (same_audio) to confirm the trace is of the judged clip.

    python bench/attn_omnivoice.py --out out [--settings base e2] [--ids en_1 en_2 ...]

Writes <out>/attn/omnivoice_<setting>.pt (dict id -> arrays).
"""
import argparse
from pathlib import Path

import run_omnivoice
import superbench as sb


class Recorder:
    """Attention function for the LLM: on every call while .q is set, records for row 0 the attention mass of the
    target frames .q on the key positions .keys, natural and as used; the output is unchanged."""

    NAME = "tagguide_attn_trace"

    def __init__(self, llm, torch):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

        self.torch, self.repeat_kv = torch, repeat_kv
        self.inner = ALL_ATTENTION_FUNCTIONS.get_interface(llm.config._attn_implementation, eager_attention_forward)
        self.q, self.keys, self.rows = None, None, []
        AttentionInterface.register(self.NAME, self)
        llm.set_attn_implementation(self.NAME)

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if self.q is not None:
            q = query[0:1, :, self.q, :].float()
            k = self.repeat_kv(key[0:1], module.num_key_value_groups).float()
            score = (q @ k.transpose(-1, -2)) * module.scaling  # [1, H, T, L]
            mask = attention_mask[0:1, :, self.q, :k.size(-2)]
            if mask.dtype == self.torch.bool:
                used = natural = score.masked_fill(~mask, -float("inf"))
            else:
                used = score + mask.float()
                natural = score + mask.float().clamp(max=0)  # the excitation bias is the only positive entry
            self.rows.append(self.torch.stack([p.softmax(-1).mean(1)[0][:, self.keys].sum(-1)
                                               for p in (natural, used)]))  # [2, T]
        return self.inner(module, query, key, value, attention_mask, **kwargs)


def same_audio(audio, path, torch, sf):
    """True if `audio` reproduces the saved clip (same length, correlation > 0.99)."""
    a = torch.as_tensor(audio).squeeze().float().cpu()
    saved = torch.tensor(sf.read(path)[0], dtype=torch.float32)
    n = min(len(a), len(saved))
    return bool(len(a) == len(saved) and torch.corrcoef(torch.stack([a[:n], saved[:n]]))[0, 1] > 0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=["base", "e2"], choices=list(sb.SETTINGS))
    ap.add_argument("--ids", nargs="*", default=None, help="items to trace (default: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    items = [it for it in sb.items("omnivoice") if args.ids is None or it["id"] in args.ids]
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    rec = Recorder(model.llm, torch)
    n_layers = model.llm.config.num_hidden_layers
    state = {}
    window, positions = guide.window, guide.tag_token_positions

    def traced_window(task, text):
        state["win"] = window(task, text)
        return state["win"]

    def traced_positions(task, text, cond_len, grouped=False):
        keys = positions(task, text, cond_len, grouped)
        t_len = task.target_lens[0]
        rec.q = torch.arange(cond_len - t_len, cond_len, device=model.device)
        rec.keys = torch.tensor(keys, device=model.device)
        return keys

    guide.window, guide.tag_token_positions = traced_window, traced_positions
    (args.out / "attn").mkdir(exist_ok=True)
    for name in args.settings:
        guide.setting = sb.SETTINGS[name]
        result = {}
        for k, it in enumerate(items):
            rec.q, rec.rows = None, []
            torch.manual_seed(args.seed)
            audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
            rec.q = None
            mass = torch.stack(rec.rows)  # [steps * layers, 2, T]
            mass = mass.view(-1, n_layers, 2, mass.shape[-1])  # [steps, layers, 2, T]
            win = state["win"].to(mass.device)
            same = same_audio(audio, args.out / f"omnivoice_{name}" / f"{it['id']}.wav", torch, sf)
            result[it["id"]] = {
                "in_natural": mass[:, :, 0, win].mean(-1).cpu(), "in_excited": mass[:, :, 1, win].mean(-1).cpu(),
                "out_natural": mass[:, :, 0, ~win].mean(-1).cpu(),
                "frames_natural": mass[:, :, 0, :].mean(1).cpu(),  # per frame, for a peak measure
                "window": [int(win.nonzero().min()), int(win.nonzero().max())], "t_len": int(win.numel()),
                "K": args.steps, "type": it["type"], "same_audio": same,
            }
            print(f"[{name} {k + 1}/{len(items)}] {it['id']} same_audio={same}", flush=True)
        torch.save(result, args.out / "attn" / f"omnivoice_{name}.pt")


if __name__ == "__main__":
    main()
