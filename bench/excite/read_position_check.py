"""Which CosyVoice 2 LM layers track the text: the evidence for reading the excitation window from layers 11 and 16.

Synthesizes a few items without guidance and records, at every decoding step, each layer's attention over the
target text (mean over heads). A layer that reads the text in order has its attention peak move forward with the
step: the Spearman correlation between the peak's text position and the step index is near 1. Reports it per layer,
for the mean over all layers, and for the mean over --layers (the read position the method uses).

    python -m bench.excite.read_position_check --items 4

Writes <out>/read_position_check.md and prints it.
"""
import argparse
from pathlib import Path

import torch

from bench import run_cosyvoice
from bench import superbench as sb


def spearman(x, y):
    def rank(v):
        v = torch.as_tensor(v, dtype=torch.float64)
        order = v.argsort()
        r = torch.empty_like(v)
        r[order] = torch.arange(len(v), dtype=torch.float64)
        for val in v.unique():  # average ranks over ties
            tie = v == val
            r[tie] = r[tie].mean()
        return r

    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--items", type=int, default=4, help="first item of each of the first N types")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_cosyvoice.add_args(ap)
    args = ap.parse_args()

    items = sb.items("cosyvoice", per_type=1)[:args.items]
    cosy = run_cosyvoice.Cosy(args)
    n_layers = cosy.lm.llm.model.config.num_hidden_layers
    rows, per_layer = [], {layer: [] for layer in range(n_layers)}
    for it in items:
        job = cosy.job(it, sb.SETTINGS["base"])
        reads = []

        def wrap(inference):
            def traced(*a, **kw):
                for tok in inference(*a, **kw):
                    if cosy.reader.mass:
                        reads.append(torch.stack(cosy.reader.mass).float().cpu())  # [layers, text tokens]
                    yield tok
            return traced

        torch.manual_seed(args.seed)
        cosy.synth(it, job, wrap)
        mass = torch.stack([m for m in reads if m.shape[0] == n_layers])  # [steps, layers, text]
        steps = list(range(mass.shape[0]))
        for layer in range(n_layers):
            per_layer[layer].append(spearman(mass[:, layer].argmax(-1), steps))
        rho_all = spearman(mass.mean(1).argmax(-1), steps)
        rho_sel = spearman(mass[:, args.layers].mean(1).argmax(-1), steps)
        rows.append(f"| {it['id']} | {mass.shape[-1]} | {len(steps)} | {rho_all:.2f} | {rho_sel:.2f} |")
        print(it["id"], f"all layers {rho_all:.2f}, layers {args.layers} {rho_sel:.2f}", flush=True)
    lines = ["# Read position of CosyVoice 2's LM", "",
             "Spearman correlation between the attention peak's position in the target text and the decoding step.",
             "", f"| item | text tokens | steps | mean of all layers | mean of layers {args.layers} |",
             "|---|---|---|---|---|", *rows, "", "| layer | min over items | mean over items |", "|---|---|---|"]
    for layer, v in per_layer.items():
        v = [x for x in v if x == x]  # nan: the peak never moves on that item
        lines.append(f"| {layer} | {min(v):.2f} | {sum(v) / len(v):.2f} |" if v else f"| {layer} | - | - |")
    text = "\n".join(lines) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "read_position_check.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
