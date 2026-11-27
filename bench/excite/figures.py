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
from bench.guidance.figures import BLUE, FPS, GRID, INK, INK2, panel, plt, save


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
        panel(ax, "abc"[k])
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


def fig_layers(out, judge="qwen3.8-omni-flash", layers=(8, 12, 13)):
    """Layer x step maps of the window frames' own attention on the tag (in_natural, [step, layer] per item).
    Left: AUC of hit against missed items with no guidance (above 0.5: hits attend more). Right: on the missed
    items, the median ratio of that attention with excitation in steps 1-K to without (bias taken out)."""
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    base = torch.load(out / "attn" / "omnivoice_base.pt", weights_only=False)
    e2 = torch.load(out / "attn" / "omnivoice_e2.pt", weights_only=False)
    hit = hits(out / "nvv_pred" / judge / "run1", "omnivoice_base")
    ids = [i for i in base if base[i]["same_audio"] and i in e2 and e2[i]["same_audio"] and i in hit]
    ok, miss = [i for i in ids if hit[i]], [i for i in ids if not hit[i]]

    def stack(trace, items):
        return torch.stack([trace[i]["in_natural"].float() for i in items]).numpy()  # [items, steps, layers]

    b_ok, b_miss, e_miss = stack(base, ok), stack(base, miss), stack(e2, miss)
    n_steps, n_layers = b_ok.shape[1:]
    a = np.array([[auc(b_ok[:, s, l].tolist(), b_miss[:, s, l].tolist()) for s in range(n_steps)]
                  for l in range(n_layers)])
    ratio = np.median(e_miss / b_miss, axis=0).T  # [layers, steps]
    K = int(base[ids[0]]["K"])

    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.04, 0.36, 1, 0.04], wspace=0.08)
    ext = [0.5, n_steps + 0.5, -0.5, n_layers - 0.5]
    panels = [(gs[0, 0], gs[0, 1], a, TwoSlopeNorm(0.5, 0.2, 0.8), "AUC, hit vs missed",
               f"(a) no guidance: hits vs misses ({len(ok)} / {len(miss)})"),
              (gs[0, 3], gs[0, 4], ratio, LogNorm(0.5, 2.0), "excited / no guidance",
               f"(b) missed items: excited in steps 1-{K}")]
    for k, (cell, ccell, z, norm, clabel, title) in enumerate(panels):
        ax = fig.add_subplot(cell)
        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower", extent=ext,
                       interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("decoding step")
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_ylabel("layer" if k == 0 else "")
        ax.set_yticks(range(0, n_layers, 4))
        ax.plot([n_steps + 0.9] * len(layers), layers, "<", color=INK, ms=3.5, clip_on=False)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        if k == 1:
            ax.axvline(K + 0.5, color=INK, lw=1, ls="--")
            ax.text(K / 2 + 0.5, n_layers - 1.2, "bias on", ha="center", va="top", fontsize=7, color=INK)
        ax.grid(False)
        cb = fig.colorbar(im, cax=fig.add_subplot(ccell))
        cb.set_label(clabel, fontsize=7)
        cb.ax.tick_params(labelsize=6.5)
        if k == 1:
            cb.ax.minorticks_off()
            cb.set_ticks([0.5, 0.71, 1, 1.41, 2], labels=["0.5", "0.7", "1", "1.4", "2"])
    save(fig, out, "fig_layers")

    rows = np.array(layers)
    other = np.setdiff1d(np.arange(n_layers), rows)
    b_late = b_miss[:, K:][:, :, rows].mean((1, 2))
    e_late = e_miss[:, K:][:, :, rows].mean((1, 2))
    up = int((e_late > b_late).sum())
    return {"n_hit": len(ok), "n_miss": len(miss),
            f"AUC layers {list(layers)}, step 1 / 16 / {n_steps}": tuple(round(float(a[rows, s].mean()), 2)
                                                                      for s in (0, 15, -1)),
            f"AUC other layers, step 1 / {n_steps}": (round(float(a[other, 0].mean()), 2),
                                                      round(float(a[other, -1].mean()), 2)),
            f"missed, layers {list(layers)}, steps {K + 1}-{n_steps}: median ratio": round(float(np.median(e_late / b_late)), 2),
            "excited above / below base": (up, len(miss) - up, sign_test(up, len(miss) - up)),
            "max ratio layer": int(ratio[:, K:].mean(1).argmax()),
            "max ratio": round(float(ratio[:, K:].mean(1).max()), 2)}


def tag_share(r, step, key):
    """Each target frame's share (%) of its attention on the target text that goes to the tag's tokens."""
    g = r["steps"][step][key].float().numpy()
    g = g / g.sum(1, keepdims=True)
    tag = set(r["tag"])
    return 100 * g[:, [k for k, (pos, _) in enumerate(r["text"]) if pos in tag]].sum(1)


def fig_neglect(out, probe="omnivoice_attn_probe_missed.pt", bins_per_window=12, span=1.0):
    """The tags OmniVoice misses, over decoding (mean over layers and heads). Frames are placed relative to their
    window (0 = window start, 1 = window end, `span` window lengths on each side) and averaged over items.
    (a) no guidance; (b) excitation in steps 1-K, the attention used (with the bias); (c) the window frames' own
    attention (bias taken out), median over items. From `bench.excite.probe_attn_omnivoice --at 1 ... 32`."""
    p = torch.load(out / "attn" / probe, weights_only=False)
    ids = [i for i in p if all(p[i][s]["same_audio"] for s in ("base", "e2"))]
    n_steps = max(p[ids[0]]["base"]["steps"])
    edges = np.linspace(-span, 1 + span, int(bins_per_window * (1 + 2 * span)) + 1)

    def binned(r, key):  # [bins, steps], frames relative to the window
        w = np.flatnonzero(r["window"].numpy())
        u = (np.arange(len(r["window"])) - w[0]) / (w[-1] + 1 - w[0])
        idx = np.digitize(u, edges) - 1
        z = np.full((len(edges) - 1, n_steps), np.nan)
        for s in range(n_steps):
            share = tag_share(r, s + 1, key)
            for b in range(len(edges) - 1):
                if (idx == b).any():
                    z[b, s] = share[idx == b].mean()
        return z

    def window_mean(r, key):
        w = r["window"].numpy().astype(bool)
        return [float(tag_share(r, s + 1, key)[w].mean()) for s in range(n_steps)]

    maps = {name: np.nanmean([binned(p[i][setting], key) for i in ids], axis=0)
            for name, setting, key in (("base", "base", "grid_natural"), ("e2", "e2", "grid_used"))}
    own = {s: np.array([window_mean(p[i][s], "grid_natural") for i in ids]) for s in ("base", "e2")}
    used = np.array([window_mean(p[i]["e2"], "grid_used") for i in ids])
    K = int(np.flatnonzero(np.median(used - own["e2"], 0) > 1e-3).max()) + 1  # steps the bias acted in

    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.5, 1.15], wspace=0.12)
    vmin, vmax = (float(np.nanpercentile(maps["base"], q)) for q in (1, 99))
    ext = [0.5, n_steps + 0.5, edges[0], edges[-1]]
    for k, (name, title) in enumerate((("base", "a"), ("e2", "b"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(maps[name], aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, extent=ext,
                       interpolation="nearest")
        for y in (0, 1):
            ax.axhline(y, color="white", lw=0.8, ls="--")
        ax.set_xticks([1, 8, 16, 24, 32])
        ax.set_xlabel("decoding step")
        panel(ax, title)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["window\nstart", "window\nend"] if k == 0 else [], fontsize=7)
        ax.grid(False)
        if name == "e2":
            ax.text((K + 1) / 2, 1 + span * 0.85, "bias on", color="white", ha="center", va="top", fontsize=7)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]), extend="both")
    cb.set_label("tag's share of text attention (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    ax = fig.add_subplot(gs[0, 4])
    x = np.arange(1, n_steps + 1)
    ax.axvspan(0.5, K + 0.5, color=GRID, lw=0)
    for s, color, label in (("base", INK2, "no guidance"), ("e2", BLUE, f"excited in steps 1-{K}")):
        ax.plot(x, np.median(own[s], 0), color=color, lw=2, label=label)
    ax.text((K + 1) / 2, ax.get_ylim()[0], f"bias on\n(used:\n{np.median(used[:, :K]):.0f}%)", ha="center",
            va="bottom", fontsize=7, color=INK2)
    ax.set_xlabel("decoding step")
    panel(ax, "c")
    ax.set_ylabel("tag's share (%)", fontsize=7.5)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    save(fig, out, "fig_neglect")

    late = slice(K, n_steps)
    up = int((own["e2"][:, late].mean(1) > own["base"][:, late].mean(1)).sum())
    return {"items": len(ids), "K": K,
            "no guidance, step 1 / 16 / 32": [round(float(np.median(own["base"][:, s])), 1) for s in (0, 15, -1)],
            "excited, used, steps 1-K": round(float(np.median(used[:, :K])), 1),
            "excited, own, step 1 / 16 / 32": [round(float(np.median(own["e2"][:, s])), 1) for s in (0, 15, -1)],
            "own ratio excited / none, steps K+1-32 (median)": round(float(np.median(
                own["e2"][:, late].mean(1) / own["base"][:, late].mean(1))), 2),
            "excited above / below": (up, len(ids) - up, sign_test(up, len(ids) - up))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(fig_attention(args.out))
    print(neglect_stats(args.out))
    print(fig_layers(args.out))


if __name__ == "__main__":
    main()
