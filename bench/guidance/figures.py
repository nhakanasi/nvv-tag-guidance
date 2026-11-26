"""Figures for the presence-guidance motivation (bench/guidance/README.md, section 2). Locally, no judge calls.

  fig_detector   The model's own tag evidence. Per frame, KL(p_tag || p_notag) between OmniVoice's predictions for
                 the text with and without the tag (first 8 steps, mean over codebooks): the expected log-likelihood
                 ratio, i.e. the expected log Bayes factor of the model's implicit tag detector. (a) along one hit
                 and one missed utterance, window shaded; (b) its mean in the window, hits against misses.
  fig_amplify    What presence guidance does to that evidence. (a) the guided probability of a tag-like token,
                 q' = q r^l1 / (q r^l1 + (1-q) ((1-q)/(1-q0))^l1) with r = q / q0, for several r at l1 = 2;
                 (b) CosyVoice 2 fixed / broken items, split by whether the read position ever reaches the tag.

    python -m bench.guidance.figures --out results

Writes <out>/figs/fig_detector.{png,pdf} and <out>/figs/fig_amplify.{png,pdf}.
"""
import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench.guidance.confusion_report import item_kl, mannwhitney_p  # noqa: E402
from bench.guidance.lcg_report import cosyvoice_tag_spans  # noqa: E402
from bench.stats import Judged, sign_test  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # sequential blue, light to dark
FPS = 25  # OmniVoice frames per second

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False,
})


def save(fig, out, name):
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figs / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_detector(out, judged):
    res = torch.load(out / "attn" / "omnivoice_confusion.pt", weights_only=False)
    hits = judged.hits("omnivoice_base")
    kept = {i: r for i, r in res.items() if r.get("same_audio") and i in hits}
    score = {i: item_kl("omnivoice", r)["notag"] for i, r in kept.items()}
    hit = {i: s for i, s in score.items() if hits[i]}
    miss = {i: s for i, s in score.items() if not hits[i]}

    def nearest(group):  # the item closest to its group's median: a typical one, not a picked one
        m = statistics.median(group.values())
        return min(group, key=lambda i: abs(group[i] - m))

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], hspace=0.55, wspace=0.3)
    top = None
    for row, (label, group, color) in enumerate((("hit", hit, BLUE), ("missed", miss, ORANGE))):
        i = nearest(group)
        r = kept[i]
        kl = r["kl"][:8, 0].float().mean(0).numpy()
        t = np.arange(len(kl)) / FPS
        ax = fig.add_subplot(gs[row, 0], sharex=top, sharey=top)
        top = top or ax
        w = np.flatnonzero(r["window"].numpy())
        ax.axvspan(w[0] / FPS, (w[-1] + 1) / FPS, color=GRID, lw=0, zorder=0)
        ax.plot(t, kl, color=color, lw=1.6)
        ax.set_title(f"{label} item ({i}): window mean {group[i]:.2f}", loc="left", fontsize=8.5, color=INK)
        ax.set_ylabel("KL per frame", fontsize=8)
        if row == 0:
            ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GRID)], labels=["tag window"], fontsize=7.5,
                      loc="upper left")
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("time (s)")
    ax = fig.add_subplot(gs[:, 1])
    rng = np.random.default_rng(0)
    for x, (label, group, color) in enumerate((("hit", hit, BLUE), ("missed", miss, ORANGE))):
        v = np.array(list(group.values()))
        ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=14, color=color, alpha=0.7, lw=0)
        m = float(np.median(v))
        ax.plot([x - 0.3, x + 0.3], [m, m], color=INK, lw=1.6)
        ax.text(x + 0.33, m, f"{m:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_xticks([0, 1], [f"hit\n(n={len(hit)})", f"missed\n(n={len(miss)})"])
    ax.set_xlim(-0.6, 1.8)
    ax.set_ylabel("mean KL in window")
    ax.set_title(f"Mann-Whitney p = {mannwhitney_p(list(hit.values()), list(miss.values())):.3f}",
                 loc="left", fontsize=8.5, color=INK)
    ax.grid(axis="x", visible=False)
    fig.suptitle("OmniVoice's own tag evidence: KL(p with tag || p without tag), first 8 steps",
                 x=0.0, y=1.06, ha="left", fontsize=9.5, color=INK)
    save(fig, out, "fig_detector")


def guided(q, r, l1=2.0):
    q0 = q / r
    a = q * r ** l1
    return a / (a + (1 - q) * ((1 - q) / (1 - q0)) ** l1)


def fig_amplify(out, judged):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.35})
    q = np.linspace(0.001, 0.6, 300)
    ax.plot(q, q, color=INK2, lw=1.2, ls="--", label="r = 1 (no change)")
    for r, color in zip((1.5, 2, 3, 4), RAMP):
        ax.plot(q, guided(q, r), color=color, lw=1.8, label=f"r = {r:g}")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.scatter([0.10], [guided(0.10, 2)], s=36, color=SURFACE, edgecolor=INK, lw=1.2, zorder=5)
    ax.annotate(f"q = 0.10, q0 = 0.05\nq' = {guided(0.10, 2):.2f}", (0.10, guided(0.10, 2)), (0.13, 0.55),
                fontsize=7.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("q: probability of a tag-like token, with the tag")
    ax.set_ylabel("q': with presence guidance")
    ax.set_title("(a) l1 = 2, r = q / q0 (q0: without the tag)", loc="left", fontsize=8.5, color=INK)

    trace = torch.load(out / "attn" / "cosyvoice_base.pt", weights_only=False)
    spans = cosyvoice_tag_spans(out)
    base, p2 = judged.hits("cosyvoice_base"), judged.hits("cosyvoice_p2")
    groups = {"reads the tag": [], "never reads it": []}
    for i in trace:
        if i in base and i in p2:
            reached = any(spans[i][0] <= int(p) < spans[i][1] for p in trace[i]["pos"])
            groups["reads the tag" if reached else "never reads it"].append(i)
    x = np.arange(len(groups))
    for k, (label, ids) in enumerate(groups.items()):
        fixed = sum(not base[i] and p2[i] for i in ids)
        broken = sum(base[i] and not p2[i] for i in ids)
        for dx, v, color in ((-0.19, fixed, BLUE), (0.19, broken, ORANGE)):
            bx.bar(k + dx, v, width=0.36, color=color, edgecolor=SURFACE, lw=2)
            bx.text(k + dx, v + 1, str(v), ha="center", va="bottom", fontsize=7.5, color=INK)
        bx.text(k, max(fixed, broken) + 9, f"p = {sign_test(fixed, broken):.3f}", ha="center", fontsize=7.5,
                color=INK2)
    bx.set_xticks(x, [f"{label}\n(n={len(ids)})" for label, ids in groups.items()])
    bx.set_ylabel("items")
    bx.set_ylim(0, 86)
    bx.grid(axis="x", visible=False)
    bx.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["fixed by guidance", "broken by guidance"], fontsize=7.5, loc="upper right")
    bx.set_title("(b) CosyVoice 2, presence l1 = 2", loc="left", fontsize=8.5, color=INK)
    save(fig, out, "fig_amplify")


def fig_frame(out, item="en_1428", cb=0, frame=14):
    """(a) One window frame of a missed item at step 0: the token distribution without the tag, with it, and with
    presence guidance. (b) Over the items whose trace reproduces the base clip: the top-token probability
    (confidence) at step 0 in the window, without the tag, with it and with guidance, and outside the window."""
    res = torch.load(out / "attn" / "omnivoice_probe.pt", weights_only=False)
    notag, base, guided = res[item]["dist"].float()[:, cb, frame]
    default = int(notag.argmax())
    order = torch.argsort(guided, descending=True)[:6].tolist()
    toks = [default] + [t for t in order if t != default]
    toks += [t for t in torch.argsort(base, descending=True)[:8].tolist() if t not in toks][:9 - len(toks)]
    raised = base > notag
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.35, 1.3], wspace=0.2)
    x = np.arange(len(toks))
    for k, (dist, title) in enumerate(((notag, "without the tag"), (base, "with the tag"),
                                       (guided, "presence guidance"))):
        ax = fig.add_subplot(gs[0, k])
        colors = [ORANGE if t == default else BLUE if raised[t] else "#b8b7b1" for t in toks]
        ax.bar(x, [float(dist[t]) for t in toks], color=colors, width=0.8, edgecolor=SURFACE, lw=1)
        top = int(dist.argmax())
        ax.text(0.97, 0.95, f"top: {float(dist.max()):.2f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK)
        ax.set_ylim(0, 0.9)
        ax.set_xticks(x, [str(t) for t in toks], rotation=90, fontsize=6.5)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_ylabel("probability")
            ax.set_xlabel("audio token (codebook 1)", fontsize=7.5)
        else:
            ax.set_yticklabels([])
        if top not in toks:
            ax.text(0.5, 0.8, f"top token {top}", transform=ax.transAxes, fontsize=7, color=INK2)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (ORANGE, BLUE)],
               labels=["default token (top without the tag)", "tokens the tag raised"], fontsize=7.5,
               loc="upper left", bbox_to_anchor=(0.08, -0.06), ncol=2, frameon=False)

    keep = [r for r in res.values() if r["same_audio"]]
    ax = fig.add_subplot(gs[0, 4])
    cols = [("no tag", [r["steps"][0]["conf"][0] for r in keep], ORANGE),
            ("tag", [r["steps"][0]["conf"][1] for r in keep], "#b8b7b1"),
            ("guided", [r["steps"][0]["conf"][2] for r in keep], BLUE),
            ("outside\nwindow", [r["steps"][0]["conf_outside"] for r in keep], INK2)]
    rng = np.random.default_rng(0)
    for k, (label, v, color) in enumerate(cols):
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(v)), v, s=8, color=color, alpha=0.6, lw=0)
        m = float(np.median(v))
        ax.plot([k - 0.28, k + 0.28], [m, m], color=INK, lw=1.6)
        ax.text(k, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(cols)), [c[0] for c in cols], fontsize=7)
    ax.set_ylabel("top-token probability")
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"window, all items (n={len(keep)})", loc="left", fontsize=8.5, color=INK)
    fig.suptitle(f"The tag removes the default but does not pick a sound; guidance picks one "
                 f"({item}, missed; window frame, step 1)", x=0.0, y=1.02, ha="left", fontsize=9.5, color=INK)
    save(fig, out, "fig_frame")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    judged = Judged(args.out)
    fig_detector(args.out, judged)
    fig_amplify(args.out, judged)
    if (args.out / "attn" / "omnivoice_probe.pt").exists():
        fig_frame(args.out)


if __name__ == "__main__":
    main()
