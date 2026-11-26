"""Figure for the excitation motivation (bench/excite/README.md, section 1). Locally, no judge calls.

  fig_attention  Where each audio frame around the tag reads in the text (en_1428, missed): a frames x text tokens
                 grid, each frame's attention on the target text normalized to 100%, at step 1 and step 16 with no
                 guidance, and at step 16 after excitation (b = 2) in steps 1-8 (the model's own attention, no bias
                 at that step); from `bench.excite.probe_attn_omnivoice`.
  neglect_stats  Over the items whose trace reproduces the judged clip: the window frames' attention on the tag's
                 tokens at step 1 and step 16, for items the base model voices (hit) and misses, and for the missed
                 items with excitation; from `bench.excite.attn_omnivoice` (omnivoice_base.pt, omnivoice_e2.pt) and
                 the judged predictions. Printed, not plotted.

    python -m bench.excite.figures --out results

Writes <out>/figs/fig_attention.{png,pdf}.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from bench.excite.attn_report import auc, hits
from bench.stats import sign_test
from bench.guidance.figures import BLUE, FPS, INK, INK2, plt, save


def fig_attention(out, item="en_1428", late=16, margin=40):
    probe = torch.load(out / "attn" / "omnivoice_attn_probe.pt", weights_only=False)[item]
    ref = probe["base"]
    tag = set(ref["tag"])
    text = ref["text"]
    first = min(k for k, (pos, _) in enumerate(text) if pos in tag)
    last = max(k for k, (pos, _) in enumerate(text) if pos in tag)
    shown = list(range(max(0, first - 10), min(len(text), last + 7)))
    tcols = [k for k in range(len(text)) if text[k][0] in tag]
    win = ref["window"].numpy().astype(bool)
    w0, w1 = int(np.flatnonzero(win)[0]), int(np.flatnonzero(win)[-1]) + 1
    f0, f1 = max(0, w0 - margin), min(len(win), w1 + margin)

    def grid(setting, step):
        g = probe[setting]["steps"][step]["grid_natural"].float().numpy()
        g = g / g.sum(1, keepdims=True)  # each frame's attention on the target text, as 100%
        tag_share = 100 * float(g[win][:, tcols].sum(1).mean())
        return 100 * g[f0:f1][:, shown], tag_share

    panels = [(("no guidance", 1), grid("base", 1)), (("no guidance", late), grid("base", late)),
              (("excited in steps 1-8", late), grid("e2", late))]
    vmax = float(np.percentile(np.concatenate([g.ravel() for _, (g, _) in panels]), 99.5))

    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.1)
    x = np.arange(len(shown))
    for k, ((title, step), (g, tag_share)) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest",
                       origin="lower", extent=[-0.5, len(shown) - 0.5, f0 / FPS, f1 / FPS])
        for y in (w0, w1):
            ax.axhline(y / FPS, color="white", lw=1, ls="--")
        ax.set_title(f"{title}\nstep {step}: tag {tag_share:.0f}%", loc="left", fontsize=8.5, color=INK)
        ax.set_xticks(x, [text[j][1].strip() or "_" for j in shown], rotation=90, fontsize=6.5)
        for lab, j in zip(ax.get_xticklabels(), shown):
            lab.set_color(BLUE if j in tcols else INK2)
            lab.set_fontweight("bold" if j in tcols else "normal")
        ax.grid(False)
        if k == 0:
            ax.set_ylabel("audio frame (s)")
        else:
            ax.set_yticklabels([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
    cb.set_label("share of the frame's text attention (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    save(fig, out, "fig_attention")
    return {"tag_share": [round(t, 1) for _, (_, t) in panels]}


def neglect_stats(out, late=16, judge="qwen3.8-omni-flash"):
    base = torch.load(out / "attn" / "omnivoice_base.pt", weights_only=False)
    e2 = torch.load(out / "attn" / "omnivoice_e2.pt", weights_only=False)
    hit = hits(out / "nvv_pred" / judge / "run1", "omnivoice_base")
    ids = [i for i in base if base[i]["same_audio"] and i in e2 and e2[i]["same_audio"] and i in hit]
    ok, miss = [i for i in ids if hit[i]], [i for i in ids if not hit[i]]

    def share(trace, i, step):  # the model's own attention, mean over layers
        return 100 * float(trace[i]["in_natural"][step - 1].float().mean())

    stats = {"n_hit": len(ok), "n_miss": len(miss)}
    for step in (1, late):
        h, m, x = ([share(t, i, step) for i in items] for t, items in ((base, ok), (base, miss), (e2, miss)))
        stats[f"step {step}"] = {"hit": round(float(np.median(h)), 2), "miss": round(float(np.median(m)), 2),
                                 "miss excited": round(float(np.median(x)), 2), "AUC hit vs miss": round(auc(h, m), 2)}
    for name, items in (("hit", ok), ("miss", miss)):
        up = sum(share(base, i, late) > share(base, i, 1) for i in items)
        down = sum(share(base, i, late) < share(base, i, 1) for i in items)
        stats[f"{name}: up / down, step 1 to {late}"] = (up, down, sign_test(up, down))
    up = sum(share(e2, i, late) > share(base, i, late) for i in miss)
    down = sum(share(e2, i, late) < share(base, i, late) for i in miss)
    stats[f"miss: excited above / below base, step {late}"] = (up, down, sign_test(up, down))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(fig_attention(args.out))
    print(neglect_stats(args.out))


if __name__ == "__main__":
    main()
