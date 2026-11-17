"""Join the attention traces (attn_omnivoice.py, attn_cosyvoice.py) with the verifier's outcomes for base and e2.

Per item, the attention mass on the tag's tokens, mean over layers, during the steps excitation acts in:
OmniVoice, the first K steps and the window's frames; CosyVoice 2, the steps whose read position is at the tag.

  base nat   the base model's own attention (b = 0)
  e2 nat     the e2 run's own attention, bias taken out
  e2 exc     the e2 run's attention as used

Items are grouped by outcome: fixed (base not hit, e2 hit), kept (hit in both), broken (hit in base only), still
(hit in neither). AUC is the probability that a random base-hit item has more base attention on its tag than a
random base-miss item (0.5: no relation; > 0.5: misses neglect the tag). The layer table gives, per layer, the
base attention of hit vs miss items and that AUC, to find the layers where neglect shows.

    python -m bench.attn_report --out out

Writes <out>/attn/attn_report.md and prints it.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch

from bench.outcomes import outcome


def hits(root, system, delta=2):
    preds = json.loads((root / system / f"{system}.json").read_text(encoding="utf-8")).values()
    return {p["id"]: outcome(p, delta) == "hit" for p in preds}


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def per_layer(model, rec):
    """[layers] attention on the tag in the steps where excitation acts (natural and as used)."""
    if model == "omnivoice":
        k = rec["K"]
        return rec["in_natural"][:k].mean(0), rec["in_excited"][:k].mean(0)
    active = (rec["pos"] >= rec["lo"]) & (rec["pos"] < rec["hi"])
    if not active.any():
        return None, None
    return rec["natural"][active].mean(0), rec["excited"][active].mean(0)


def fmt(xs):
    return f"{statistics.median(xs):.4f}" if xs else "NA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default="qwen3.8-omni-flash")
    args = ap.parse_args()
    pred_root = args.out / "nvv_pred" / args.judge / "run1"
    lines = []
    for model in ("omnivoice", "cosyvoice"):
        base_f, e2_f = args.out / "attn" / f"{model}_base.pt", args.out / "attn" / f"{model}_e2.pt"
        if not (base_f.exists() and e2_f.exists()):
            continue
        base, e2 = torch.load(base_f), torch.load(e2_f)
        hb, he = hits(pred_root, f"{model}_base"), hits(pred_root, f"{model}_e2")
        rows, layer_rows = {}, []
        for i in base:
            bn, _ = per_layer(model, base[i])
            en, ex = per_layer(model, e2[i])
            if bn is None or en is None:
                continue
            group = {(False, True): "fixed", (True, True): "kept", (True, False): "broken",
                     (False, False): "still"}[(hb[i], he[i])]
            rows[i] = (group, float(bn.mean()), float(en.mean()), float(ex.mean()), bn)
        same = sum(base[i]["same_audio"] and e2[i]["same_audio"] for i in base)
        lines += [f"## {model}", "", f"items traced {len(base)}, same audio as judged clip {same}, "
                  f"with steps at the tag in both runs {len(rows)}", "",
                  "| group | n | base nat | e2 nat | e2 exc | exc / nat | e2 nat > base nat |", "|---|---|---|---|---|---|---|"]
        for g in ("fixed", "kept", "broken", "still"):
            r = [v for v in rows.values() if v[0] == g]
            ratio = [v[3] / v[2] for v in r if v[2] > 0]
            up = sum(v[2] > v[1] for v in r)
            lines.append(f"| {g} | {len(r)} | {fmt([v[1] for v in r])} | {fmt([v[2] for v in r])} | "
                         f"{fmt([v[3] for v in r])} | {fmt(ratio)} | {up}/{len(r)} |")
        hit = [v[1] for v in rows.values() if v[0] in ("kept", "broken")]
        miss = [v[1] for v in rows.values() if v[0] in ("fixed", "still")]
        fixed = [v[1] for v in rows.values() if v[0] == "fixed"]
        lines += ["", f"base attention, base-hit vs base-miss: AUC {auc(hit, miss):.3f} "
                  f"(median {fmt(hit)} vs {fmt(miss)}); base-hit vs fixed: AUC {auc(hit, fixed):.3f}", "",
                  "| layer | base-hit median | base-miss median | AUC |", "|---|---|---|---|"]
        n_layers = next(iter(rows.values()))[4].numel()
        for layer in range(n_layers):
            h = [float(v[4][layer]) for v in rows.values() if v[0] in ("kept", "broken")]
            m = [float(v[4][layer]) for v in rows.values() if v[0] in ("fixed", "still")]
            layer_rows.append(f"| {layer} | {fmt(h)} | {fmt(m)} | {auc(h, m):.3f} |")
        lines += layer_rows + [""]
    text = "\n".join(lines)
    (args.out / "attn" / "attn_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
