"""Where the tag's frames look, without and with excitation (OmniVoice, a few items, chosen steps).

Re-synthesizes each item with no guidance (base) and with excitation (e2), and records, at the chosen decoding steps,
the conditional pass's attention (mean over heads and layers): the model's own attention (the bias taken out) and the
attention actually used. Saved per step: from the tag window's frames to every input position (mean over the
window's frames), and from every target frame to every text token of the target text (a frames x tokens grid). Also
saves the target text's tokens and their positions, for plotting.

    python -m bench.excite.probe_attn_omnivoice --out out --ids en_1428 --at 1 8 16 24

Writes <out>/attn/omnivoice_attn_probe.pt (dict id -> setting -> {"steps": {step: natural [L], used [L],
grid_natural / grid_used [T, text tokens]}, the window, text tokens and positions, same_audio}).
"""
import argparse
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb
from bench.excite.attn_omnivoice import same_audio


class FullRecorder:
    """Attention function: while .q is set, counts the calls (one per layer per decoding step) and, at the 1-based
    steps .steps, stores row 0's attention (mean over heads), natural and as used: from the frames .q[.win] to every
    key (mean over those frames), and from every frame .q to the keys .cols."""

    NAME = "tagguide_attn_probe"

    def __init__(self, llm, torch, layers):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

        self.torch, self.repeat_kv, self.layers = torch, repeat_kv, layers
        self.inner = ALL_ATTENTION_FUNCTIONS.get_interface(llm.config._attn_implementation, eager_attention_forward)
        self.q, self.win, self.cols, self.steps = None, None, None, set()
        self.reset()
        AttentionInterface.register(self.NAME, self)
        llm.set_attn_implementation(self.NAME)

    def reset(self):
        self.calls, self.rows, self.grids = 0, {}, {}

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if self.q is not None:
            step = self.calls // self.layers + 1
            self.calls += 1
            if step in self.steps:
                q = query[0:1, :, self.q, :].float()
                k = self.repeat_kv(key[0:1], module.num_key_value_groups).float()
                score = (q @ k.transpose(-1, -2)) * module.scaling
                mask = attention_mask[0:1, :, self.q, :k.size(-2)]
                if mask.dtype == self.torch.bool:  # no excitation at this step
                    used = natural = score.masked_fill(~mask, -float("inf"))
                else:  # the excitation bias is the only positive entry
                    used, natural = score + mask.float(), score + mask.float().clamp(max=0)
                att = [p.softmax(-1).mean((0, 1)) for p in (natural, used)]  # [Q, K]
                self.rows.setdefault(step, []).append(self.torch.stack([a[self.win].mean(0) for a in att]).cpu())
                self.grids.setdefault(step, []).append(self.torch.stack([a[:, self.cols] for a in att]).cpu())
        return self.inner(module, query, key, value, attention_mask, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--at", type=int, nargs="+", default=[1, 8, 16, 24], help="1-based decoding steps")
    ap.add_argument("--settings", nargs="+", default=["base", "e2"])
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from omnivoice.models import omnivoice as ov

    items = [it for it in sb.items("omnivoice") if it["id"] in args.ids]
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    rec = FullRecorder(model.llm, torch, model.llm.config.num_hidden_layers)
    rec.steps = set(args.at)
    state = {}
    window, positions = guide.window, guide.tag_token_positions

    def traced_window(task, text):
        state["win"] = window(task, text)
        return state["win"]

    def traced_positions(task, text, cond_len, grouped=False):
        keys = positions(task, text, cond_len, grouped)
        t_len = task.target_lens[0]
        rec.q = torch.arange(cond_len - t_len, cond_len, device=model.device)
        rec.win = state["win"].to(model.device)
        wrapped = f"<|text_start|>{ov._combine_text(ref_text=task.ref_texts[0], text=text)}<|text_end|>"
        enc = model.text_tokenizer(wrapped, return_offsets_mapping=True)
        style_len = cond_len - len(enc.input_ids) - task.ref_audio_tokens[0].size(-1) - t_len
        start = wrapped.rfind(text)  # the target text, after the reference text
        state["text"] = [(style_len + i, wrapped[a:b]) for i, (a, b) in enumerate(enc.offset_mapping)
                         if b > a and a >= start]
        rec.cols = torch.tensor([pos for pos, _ in state["text"]], device=model.device)
        state["tag"], state["cond_len"] = keys, cond_len
        state["audio"] = (cond_len - t_len - task.ref_audio_tokens[0].size(-1), cond_len - t_len)  # voice prompt
        return keys

    guide.window, guide.tag_token_positions = traced_window, traced_positions
    result = {}
    for it in items:
        result[it["id"]] = {}
        for name in args.settings:
            guide.setting = sb.SETTINGS[name]
            rec.q = None
            rec.reset()
            torch.manual_seed(args.seed)
            audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
            rec.q = None
            steps = {}
            for step in sorted(rec.rows):
                rows = torch.stack(rec.rows[step]).mean(0)  # [2, L], mean over layers
                grid = torch.stack(rec.grids[step]).mean(0)  # [2, T, text tokens]
                steps[step] = {"natural": rows[0], "used": rows[1],
                               "grid_natural": grid[0].half(), "grid_used": grid[1].half()}
            r = {"steps": steps, "text": state["text"], "tag": state["tag"], "audio": state["audio"],
                 "cond_len": state["cond_len"], "window": state["win"].cpu(),
                 "same_audio": same_audio(audio, args.out / f"omnivoice_{name}" / f"{it['id']}.wav", torch, sf)}
            result[it["id"]][name] = r
            print(it["id"], name, " ".join("step %d: %.4f / %.4f" % (k, v["natural"][r["tag"]].sum(),
                                                                       v["used"][r["tag"]].sum())
                                           for k, v in steps.items()), "same_audio=%s" % r["same_audio"], flush=True)
    torch.save(result, args.out / "attn" / "omnivoice_attn_probe.pt")


if __name__ == "__main__":
    main()
