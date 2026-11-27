"""Before / after presence guidance on two missed OmniVoice laughs: mel spectrograms and FlexSED's laughter posterior.

Reads the clips <clips>/omnivoice_{base,p2}/<id>.wav, the windows (results/flexsed/omnivoice_windows.json, frames at
25 per second) and the FlexSED tracks (results/flexsed/omnivoice_tracks.json: per item and setting, the target
class's posterior "p" at hop "fs" seconds, for the 100 laugh and sigh items). The tracks were made with FlexSED
through VLSP's TagConfirmer, which is not part of this repository.

    python -m bench.guidance.fig_example --clips out --out results [--ids en_1428 en_1414]

Writes <out>/figs/fig_example.{png,pdf}.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bench.guidance.figures import BLUE, FPS, GRID, INK2, ORANGE, panel, plt, save


def mel(path):
    import librosa
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    m = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, hop_length=sr // 100)
    return librosa.power_to_db(m, ref=np.max), len(y) / sr


def fig_example(clips, out, ids=("en_1428", "en_1414")):
    windows = json.loads((out / "flexsed" / "omnivoice_windows.json").read_text())
    tracks = json.loads((out / "flexsed" / "omnivoice_tracks.json").read_text())
    fig, axes = plt.subplots(3, len(ids), figsize=(7.4, 4.6),
                             gridspec_kw={"height_ratios": [1, 1, 0.8], "hspace": 0.35, "wspace": 0.12})
    for c, i in enumerate(ids):
        a, b = (v / FPS for v in windows[i]["w"])
        for r, (s, name) in enumerate((("base", "no guidance"), ("p2", "presence guidance"))):
            ax = axes[r, c]
            m, dur = mel(clips / f"omnivoice_{s}" / f"{i}.wav")
            ax.imshow(m, origin="lower", aspect="auto", extent=[0, dur, 0, 80], cmap="magma", vmin=-70, vmax=0)
            for x in (a, b):
                ax.axvline(x, color="white", lw=1.0, ls="--")
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim(0, dur)
            ax.grid(False)
            ax.text(0.01, 0.95, name, transform=ax.transAxes, va="top", color="white", fontsize=8, weight="bold")
        ax = axes[2, c]
        ax.axvspan(a, b, color=GRID, lw=0)
        for s, color, name in (("base", ORANGE, "no guidance"), ("p2", BLUE, "presence")):
            p = np.array(tracks[i][s]["p"])
            ax.plot(np.arange(len(p)) * tracks[i][s]["fs"], p, color=color, lw=1.6, label=name)
        ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
        ax.set_xlim(0, dur)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("time (s)")
        panel(ax, "ab"[c])
        if c == 0:
            ax.set_ylabel("FlexSED P(laughter)")
            ax.legend(fontsize=7.5, loc="upper left")
            ax.text(dur * 0.99, 0.52, "detection threshold", fontsize=7, color=INK2, va="bottom", ha="right")
        else:
            ax.set_yticklabels([])
    save(fig, out, "fig_example")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="directory with omnivoice_base/ and omnivoice_p2/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", nargs=2, default=["en_1428", "en_1414"])
    args = ap.parse_args()
    fig_example(args.clips, args.out, args.ids)


if __name__ == "__main__":
    main()
