"""The model's own confusion matrix (OmniVoice): which tags it can tell apart where the tag is voiced, and how far
from the tag a tag changes the prediction at all.

On the base trajectory (no guidance, same seed and voice as the base clips), the first K steps also run one
conditional pass per variant of the text: the tag removed, and the tag swapped for every other tag of the
inventory. For each variant b and target frame, the KL divergence KL(c_a || c_b) between the conditional
log-probabilities with the asked-for tag a and with b, averaged over the codebooks. Nothing is added to the scores,
so the audio is the base clip's (same_audio).

  - Confusion matrix: mean KL on the window frames, per asked-for type and variant. The model separates a from b
    as much as the KL; the least separable other tag is the confuser it is most likely to fall back on.
  - Locality: the per-frame KL of the notag variant against the frame's distance from the window, which shows how
    many frames the tag's influence spans.

    python -m bench.guidance.confusion_omnivoice --out out

Writes <out>/attn/omnivoice_confusion.pt (dict id -> per-step, per-variant frame KLs); `bench.guidance.confusion_report`
summarizes.
"""
import argparse
import types
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb
from bench.excite.attn_omnivoice import same_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None, help="items (default: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    from tagguide import contrast
    from tagguide import omnivoice as tov

    items = [it for it in sb.items("omnivoice") if args.ids is None or it["id"] in args.ids]
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    inventory = [t.strip("[]") for t in sb.CONFIG["omnivoice"]["confuser"]]
    state = {}

    def texts(setting, text, notag, confuser):
        """The text, then the variants: no tag, then every other tag of the inventory in place of the tag."""
        tag = guide.tags.span.search(text).group(1)
        state["variants"] = ["notag"] + [t for t in inventory if t != tag]
        swapped = [guide.tags.span.sub(f"[{t}]", text) for t in state["variants"][1:]]
        return [text, notag] + swapped

    def record(lp, c, others, setting, m=None):
        """Per-frame KL(c || variant), mean over codebooks; the scores are returned unchanged."""
        p = c.exp()
        state["kl"].append(torch.stack([(p * (c - o)).sum(-1).mean(1)[0] for o in others]).half().cpu())  # [V, T]
        state["win"] = m.view(-1).bool().cpu()
        return lp

    tov.contrast = types.SimpleNamespace(texts=texts, apply=record)
    guide.setting = (1, 0, 0)  # every step < K runs all the streams; `record` adds nothing
    (args.out / "attn").mkdir(parents=True, exist_ok=True)
    result = {}
    for k, it in enumerate(items):
        state["kl"] = []
        torch.manual_seed(args.seed)
        audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
        same = same_audio(audio, args.out / "omnivoice_base" / f"{it['id']}.wav", torch, sf)
        result[it["id"]] = {"kl": torch.stack(state["kl"]), "window": state["win"], "variants": state["variants"],
                            "type": it["type"], "same_audio": same}
        inside = result[it["id"]]["kl"][:, 0, state["win"]].float().mean()
        print(f"[confusion {k + 1}/{len(items)}] {it['id']} notag KL in window {inside:.3f} same_audio={same}",
              flush=True)
    torch.save(result, args.out / "attn" / "omnivoice_confusion.pt")
    tov.contrast = contrast


if __name__ == "__main__":
    main()
