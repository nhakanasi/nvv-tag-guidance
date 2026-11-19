"""Summarizes the model-internal confusion runs (confusion_omnivoice.py, confusion_cosyvoice.py).

  1. Confusion matrix: mean KL(c_tag || c_variant) inside our window, per asked-for type (rows) and variant
     (columns: notag, then every tag of the inventory). Lower is less separable. Per row, the least separable other
     tag against the confuser configs/superbench.yaml uses.
  2. Locality: the notag KL by distance from the window (OmniVoice: frames; CosyVoice 2: read position in text
     tokens), as a share of the KL at the window.
  3. Presence KL and the base outcome: is the tag weaker (lower notag KL) on the items the base model misses? Only
     items whose trajectory reproduced the judged base clip.

    python -m bench.guidance.confusion_report --out out

Writes <out>/confusion.md and prints it.
"""
import argparse
import statistics
from pathlib import Path

import torch

from bench import superbench as sb
from bench.stats import Judged, table


def mannwhitney_p(a, b):
    """Two-sided Mann-Whitney U p-value, normal approximation (no tie correction)."""
    n1, n2 = len(a), len(b)
    if not n1 or not n2:
        return float("nan")
    ranked = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    r1 = sum(i + 1 for i, (_, g) in enumerate(ranked) if g == 0)
    u = r1 - n1 * (n1 + 1) / 2
    mu, sd = n1 * n2 / 2, (n1 * n2 * (n1 + n2 + 1) / 12) ** 0.5
    z = abs(u - mu) / sd
    return float(torch.erfc(torch.tensor(z / 2 ** 0.5)))


def item_kl(model, r):
    """{variant: mean KL inside the window} for one item."""
    if model == "omnivoice":  # kl [steps, V, T]
        k = r["kl"][:, :, r["window"]].float().mean((0, 2))
    else:  # kl [steps, 1 + V]: step index, then KL per variant
        lo, hi = r["window"]
        steps = r["kl"][:, 0].long()
        inside = (r["pos"][steps] >= lo) & (r["pos"][steps] < hi)
        k = r["kl"][inside, 1:].mean(0) if inside.any() else torch.full((len(r["variants"]),), float("nan"))
    return dict(zip(r["variants"], k.tolist()))


def matrix(model, res):
    tagmap, confuser = sb.CONFIG[model]["tags"], sb.CONFIG[model]["confuser"]
    inv = list(confuser)
    if model == "omnivoice":
        inv = [t.strip("[]") for t in inv]
    cols = ["notag"] + inv
    rows, picks = [], []
    for kind, tag in tagmap.items():
        own = tag.strip("[]") if model == "omnivoice" else tag
        per = [item_kl(model, r) for r in res.values() if r["type"] == kind]
        if not per:
            continue
        mean = {c: statistics.fmean([p[c] for p in per if c in p and p[c] == p[c]] or [float("nan")]) for c in cols
                if c != own}
        least = min((c for c in mean if c != "notag"), key=lambda c: mean[c])
        rows.append([kind, len(per)] + [f"{mean[c]:.3f}" if c in mean else "-" for c in cols])
        picks.append([kind, least, confuser[tag].strip("[]") if model == "omnivoice" else confuser[tag],
                      f"{mean[least]:.3f}", f"{mean['notag']:.3f}"])
    return (table(["type", "items"] + cols, rows),
            table(["type", "least separable tag", "configured confuser", "KL to it", "KL to notag"], picks))


def locality(model, res):
    """Mean notag KL by distance bin from the window, relative to the KL inside it."""
    bins = {}
    for r in res.values():
        if model == "omnivoice":
            kl = r["kl"][:, 0].float().mean(0)  # [T]
            idx = torch.nonzero(r["window"]).squeeze(-1)
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            pos = torch.arange(len(kl))
        else:
            steps = r["kl"][:, 0].long()
            kl, pos = r["kl"][:, 1], r["pos"][steps]
            lo, hi = r["window"]
        dist = torch.where(pos < lo, pos - lo, torch.where(pos >= hi, pos - hi + 1, torch.zeros_like(pos)))
        for d, v in zip(dist.tolist(), kl.tolist()):
            bins.setdefault(d, []).append(v)
    edges = ([(-200, -76), (-75, -51), (-50, -26), (-25, -1), (0, 0), (1, 25), (26, 50), (51, 75), (76, 200)]
             if model == "omnivoice" else [(-50, -9), (-8, -5), (-4, -1), (0, 0), (1, 4), (5, 8), (9, 50)])
    inside = statistics.fmean(bins.get(0, [float("nan")]))
    rows = []
    for a, b in edges:
        vals = [v for d in range(a, b + 1) for v in bins.get(d, [])]
        if vals:
            m = statistics.fmean(vals)
            label = "in window" if (a, b) == (0, 0) else f"{a} to {b}"
            rows.append([label, len(vals), f"{m:.3f}", f"{m / inside:.2f}"])
    unit = "frames (25/s)" if model == "omnivoice" else "read position, text tokens"
    return table([f"distance from window, {unit}", "points", "notag KL", "share of in-window"], rows)


def by_outcome(model, res, judged):
    if not judged.has(f"{model}_base"):
        return "(base not judged)"
    hits = judged.hits(f"{model}_base")
    kept = {i: r for i, r in res.items() if r.get("same_audio") and i in hits}
    hit = [item_kl(model, r)["notag"] for i, r in kept.items() if hits[i]]
    miss = [item_kl(model, r)["notag"] for i, r in kept.items() if not hits[i]]
    hit, miss = [v for v in hit if v == v], [v for v in miss if v == v]
    rows = [["hit", len(hit), f"{statistics.median(hit):.3f}" if hit else "-"],
            ["missed", len(miss), f"{statistics.median(miss):.3f}" if miss else "-"]]
    return (table(["base outcome", "items (same audio)", "median notag KL in window"], rows) +
            f"\n\nMann-Whitney p = {mannwhitney_p(hit, miss):.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    judged = Judged(args.out)
    lines = ["# The models' own confusion matrix", ""]
    for model, name in (("omnivoice", "OmniVoice"), ("cosyvoice", "CosyVoice 2")):
        path = args.out / "attn" / f"{model}_confusion.pt"
        if not path.exists():
            continue
        res = torch.load(path)
        same = sum(bool(r.get("same_audio")) for r in res.values())
        conf, picks = matrix(model, res)
        lines += [f"## {name} ({len(res)} items, {same} reproduce the base clip)", "",
                  "### Mean KL(tag || variant) in the window", "", conf, "", "### Confuser", "", picks, "",
                  "### Locality of the tag's effect (notag KL)", "", locality(model, res), "",
                  "### Presence KL and the base outcome", "", by_outcome(model, res, judged), ""]
    text = "\n".join(lines)
    (args.out / "confusion.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
