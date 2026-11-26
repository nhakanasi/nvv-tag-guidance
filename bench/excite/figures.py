"""Figure for the excitation motivation (bench/excite/README.md, section 1). Locally, no judge calls.

  fig_attention  (a) Where each audio frame around the tag looks at step 1 (en_1428, missed): a frames x text tokens
                 grid around the tag, the model's own attention and with excitation (b = 2), from
                 `bench.excite.probe_attn_omnivoice`. (b) Over the items whose trace reproduces the judged clip:
                 the attention share on the tag's tokens in steps 1-8, in the window without and with excitation,
                 and outside the window, from `bench.excite.attn_omnivoice` (omnivoice_base.pt, omnivoice_e2.pt).

    python -m bench.excite.figures --out results

Writes <out>/figs/fig_attention.{png,pdf}.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from bench.guidance.figures import BLUE, FPS, INK, INK2, ORANGE, SURFACE, plt, save

GRAY = "#b8b7b1"


def fig_attention(out, item="en_1428", before=10, after=6, margin=40):
    probe = torch.load(out / "attn" / "omnivoice_attn_probe.pt", weights_only=False)[item]
    tag = set(probe["tag"])
    text = probe["text"]
    first = next(k for k, (pos, _) in enumerate(text) if pos in tag)
    last = max(k for k, (pos, _) in enumerate(text) if pos in tag)
    shown = text[max(0, first - before):last + 1 + after]
    a0, a1 = probe["audio"]

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", [SURFACE, "#86b6ef", "#2a78d6", "#104281"])
    cols = [k for k, (pos, _) in enumerate(text) if any(pos == p for p, _ in shown)]
    win = probe["window"].numpy()
    w0, w1 = int(np.flatnonzero(win)[0]), int(np.flatnonzero(win)[-1]) + 1
    f0, f1 = max(0, w0 - margin), min(len(win), w1 + margin)
    grids = {key: 100 * probe[f"grid_{key}"].float().numpy()[f0:f1][:, cols] for key in ("natural", "used")}
    vmax = float(np.percentile(np.concatenate([g.ravel() for g in grids.values()]), 99.5))

    fig = plt.figure(figsize=(7.4, 3.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 1.3, 0.07, 0.8, 1.15], wspace=0.12)
    x = np.arange(len(shown))
    for k, (key, title) in enumerate((("natural", "model's own"), ("used", "with excitation"))):
        att = probe[key]
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(grids[key], aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest",
                       extent=[-0.5, len(shown) - 0.5, f1 / FPS, f0 / FPS])
        for y in (w0, w1):
            ax.axhline(y / FPS, color=ORANGE, lw=1, ls="--")
        share = 100 * float(att[list(tag)].sum())
        ax.set_title(f"{title}\non the tag: {share:.1f}%", loc="left", fontsize=8.5, color=INK)
        ax.set_xticks(x, [s.strip() or "_" for _, s in shown], rotation=90, fontsize=6.5)
        for lab, (p, _) in zip(ax.get_xticklabels(), shown):
            lab.set_color(BLUE if p in tag else INK2)
            lab.set_fontweight("bold" if p in tag else "normal")
        ax.grid(False)
        if k == 0:
            ax.set_ylabel("audio frame (s)")
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("attention (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    base = torch.load(out / "attn" / "omnivoice_base.pt", weights_only=False)
    e2 = torch.load(out / "attn" / "omnivoice_e2.pt", weights_only=False)
    ids = [i for i in base if base[i]["same_audio"] and i in e2 and e2[i]["same_audio"]]
    cols = [("window,\nbase", [100 * float(base[i]["in_natural"][:8].mean()) for i in ids], GRAY),
            ("window,\nexcited", [100 * float(e2[i]["in_excited"][:8].mean()) for i in ids], BLUE),
            ("outside\nwindow", [100 * float(base[i]["out_natural"][:8].mean()) for i in ids], INK2)]
    ax = fig.add_subplot(gs[0, 4])
    rng = np.random.default_rng(0)
    for k, (label, v, color) in enumerate(cols):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=color, alpha=0.6, lw=0)
        m = float(np.median(v))
        ax.plot([k - 0.28, k + 0.28], [m, m], color=INK, lw=1.6)
        ax.text(k + 0.3, m, f"{m:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7)
    ax.set_ylabel("attention on the tag, steps 1-8 (%)")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.5, 2.7)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"all items (n={len(ids)})", loc="left", fontsize=8.5, color=INK)
    fig.suptitle(f"The tag's frames give the tag about 1% of their attention; excitation raises it about 5-fold "
                 f"({item}, missed; step 1; dashed: the window)", x=0.0, y=1.02, ha="left", fontsize=9.5, color=INK)
    save(fig, out, "fig_attention")
    return {"n": len(ids), "medians": [float(np.median(v)) for _, v, _ in cols]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(fig_attention(args.out))


if __name__ == "__main__":
    main()
