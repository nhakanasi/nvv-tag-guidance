"""The model's own confusion matrix (CosyVoice 2): which tags it can tell apart, and at which read positions a tag
changes the next speech token at all.

On the base trajectory (no guidance, same seed and voice as the base clips), every step also runs one LM stream per
variant of the text: the tag removed, and the tag swapped for every other tag of the inventory, all fed the sampled
tokens. For each variant b and step, the KL divergence KL(c_a || c_b) between the next-token log-probabilities with
the asked-for tag a and with b, and the step's read position. Nothing is added to the scores.

  - Confusion matrix: mean KL over the steps in our window, per asked-for type and variant; the least separable
    other tag is the confuser the model is most likely to fall back on.
  - Locality: the notag KL against the read position's distance from the tag, in text tokens.

    python -m bench.guidance.confusion_cosyvoice --out out [--per_type 20]

Writes <out>/attn/cosyvoice_confusion.pt (dict id -> per-step KLs and read positions); `bench.guidance.confusion_report`
summarizes.
"""
import argparse
import types
from pathlib import Path

from bench import run_cosyvoice
from bench import superbench as sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_cosyvoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    from bench.excite.attn_omnivoice import same_audio
    from tagguide import contrast
    from tagguide import cosyvoice as tcv

    items = sb.items("cosyvoice", per_type=args.per_type)
    cosy = run_cosyvoice.Cosy(args)
    inventory = list(sb.CONFIG["cosyvoice"]["confuser"])
    state = {}

    def texts(setting, text, notag, confuser):
        return [text, notag] + state["swapped"]

    def record(lp, c, others, setting, m=None):
        """KL(c || variant) for this step, tagged with the step index; the scores are returned unchanged."""
        p = c.exp()
        state["kl"].append([len(state["pos"])] + [float((p * (c - o)).sum()) for o in others])
        return lp

    def gate(pos):
        state["pos"].append(pos)
        return True

    tcv.contrast = types.SimpleNamespace(texts=texts, apply=record)
    (args.out / "attn").mkdir(parents=True, exist_ok=True)
    path = args.out / "attn" / "cosyvoice_confusion.pt"
    result = torch.load(path) if path.exists() else {}
    for k, it in enumerate(items):
        if it["id"] in result:
            continue
        others = [t for t in inventory if t != it["tag"]]
        state.update(kl=[], pos=[], swapped=[cosy.encode(it["text"].replace(it["tag"], t)) for t in others])
        job = cosy.job(it, (1, 0, 0))  # all streams at every step (gate); `record` adds nothing
        job["gate"] = gate
        torch.manual_seed(args.seed)
        audio = cosy.synth(it, job)
        same = same_audio(audio, args.out / "cosyvoice_base" / f"{it['id']}.wav", torch, sf)
        lo, hi = job["window"][0] - job["dilate"], job["window"][1] + job["dilate"]
        result[it["id"]] = {"kl": torch.tensor(state["kl"]), "pos": torch.tensor(state["pos"]),
                            "variants": ["notag"] + others, "tag": job["tag"], "window": (lo, hi),
                            "type": it["type"], "same_audio": same}
        if (k + 1) % 20 == 0:
            torch.save(result, path)
        print(f"[confusion {k + 1}/{len(items)}] {it['id']} steps={len(state['pos'])} same_audio={same}", flush=True)
    torch.save(result, path)
    tcv.contrast = contrast


if __name__ == "__main__":
    main()
