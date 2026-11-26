"""What the tag does to OmniVoice's prediction at the tag's frames, and what presence guidance does to it.

On the base trajectory (no guidance; same seed and voice as the base clips), in each of the first K steps and on
each frame of the tag's window, three distributions over a codebook's tokens:

    without the tag    lp + (1 + g) (c_notag - c)      the model's score had the text no tag (g: OmniVoice's CFG)
    with the tag       lp = c + g (c - u)              what the base model decodes from
    presence guidance  lp + l1 (c - c_notag)           what `p2` decodes from on this state

OmniVoice decodes each frame's token as the argmax. Per item: the share of window frames x codebooks where the tag
leaves the top token unchanged, where guidance changes it, the total variation between the distributions with
and without the tag (how much probability the tag moves), and the confidence (top-token probability) of each, against
the base confidence outside the window (OmniVoice commits the most confident frames first). For the --dump items, the step-0 distributions on the
window frames are saved for plotting.

    python -m bench.guidance.probe_omnivoice --out out [--dump en_1428 en_1414]

Writes <out>/attn/omnivoice_probe.pt.
"""
import argparse
import types
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb
from bench.excite.attn_omnivoice import same_audio

G = 2.0  # OmniVoice's default CFG scale, used by every run here
L1 = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="items (default: all)")
    ap.add_argument("--dump", nargs="*", default=[], help="items whose step-0 distributions are saved")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    from tagguide import omnivoice as tov

    items = [it for it in sb.items("omnivoice") if args.ids is None or it["id"] in args.ids]
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    state = {}

    def record(lp, c, others, setting, m=None):
        """Statistics of the three distributions on the window frames; the scores are returned unchanged."""
        win = m.view(-1).bool()
        base = lp[0][:, win]  # [C, W, V]
        notag = base + (1 + G) * (others[0][0][:, win] - c[0][:, win])
        guided = base + L1 * (c[0][:, win] - others[0][0][:, win])
        mask_id = model.config.audio_mask_id
        p = [torch.softmax(x.index_fill(-1, torch.tensor([mask_id], device=x.device), -float("inf")), -1)
             for x in (notag, base, guided)]
        top = [x.argmax(-1) for x in p]
        state["steps"].append({"same_top": (top[0] == top[1]).float().mean().item(),
                               "guided_changes": (top[2] != top[1]).float().mean().item(),
                               "tv": 0.5 * (p[1] - p[0]).abs().sum(-1).mean().item(),
                               "conf": [x.max(-1)[0].mean().item() for x in p],  # notag, base, guided
                               "conf_outside": torch.softmax(lp[0][:, ~win], -1).max(-1)[0].mean().item()
                               if (~win).any() else float("nan")})
        if state["dump"] and len(state["steps"]) == 1:
            state["dist"] = torch.stack(p).half().cpu()  # [3, C, W, V]
        state["win"] = win.cpu()
        return lp

    tov.contrast = types.SimpleNamespace(texts=tov.contrast.texts, apply=record)
    guide.setting = (1, 0, 0)  # runs the notag stream in the first K steps; `record` adds nothing
    result = {}
    for k, it in enumerate(items):
        state.update(steps=[], dump=it["id"] in args.dump, dist=None)
        torch.manual_seed(args.seed)
        audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
        same = same_audio(audio, args.out / "omnivoice_base" / f"{it['id']}.wav", torch, sf)
        result[it["id"]] = {"steps": state["steps"], "window": state["win"], "type": it["type"],
                            "same_audio": same, "dist": state["dist"]}
        s0 = state["steps"][0]
        print(f"[probe {k + 1}/{len(items)}] {it['id']} same_top {s0['same_top']:.2f} "
              f"guided_changes {s0['guided_changes']:.2f} tv {s0['tv']:.3f} same_audio={same}", flush=True)
    (args.out / "attn").mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out / "attn" / "omnivoice_probe.pt")


if __name__ == "__main__":
    main()
