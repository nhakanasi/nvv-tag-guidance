"""Numbers behind bench/excite/README.md, recomputed from saved outputs (no synthesis, no judge calls).

  1. The tag gets little attention, and raising it voices the tag: attention shares (natural / excited) from the
     traces, e2 against no guidance, the placebo and reseed controls.
  2. The tag is settled when its frames are committed: OmniVoice's commit schedule and excitation moved to later
     steps.
  3. A window, because one tag spans many frames: tag length in tokens, window length in frames, and how often the
     CosyVoice 2 window is on.

Sections 1 (attention) and 3 (window length) need the traces of bench.excite.attn_omnivoice / attn_cosyvoice in
<out>/attn/ and torch; the tag token counts need transformers and the OmniVoice tokenizer (OMNIVOICE_CKPT). Missing
inputs skip their part. Run in the OmniVoice environment:

    python -m bench.excite.evidence --out out

Writes <out>/excite_evidence.md and prints it.
"""
import argparse
import os
import statistics
from pathlib import Path

from bench import superbench as sb
from bench.excite.control_omnivoice import CONTROLS
from bench.stats import Judged, fmt_p, paired, table

FRAME_RATE = 25            # OmniVoice audio frames per second
NUM_STEP, T_SHIFT = 32, 0.1  # OmniVoice's default decoding steps and time shift


def committed(step):
    """Share of target frames OmniVoice has committed before step `step` (0-based) of its schedule."""
    u = step / NUM_STEP
    return T_SHIFT * u / (1 + (T_SHIFT - 1) * u)


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pct(x):
    return f"{100 * x:.2f}%"


def attention_section(out, lines):
    attn = out / "attn"
    try:
        import torch

        from bench.excite.attn_report import per_layer
    except ImportError:
        lines += ["Attention shares skipped: torch is not installed.", ""]
        return None
    traces = {f.stem: torch.load(f) for f in attn.glob("*.pt")} if attn.exists() else {}
    rows = []
    ob, oe = traces.get("omnivoice_base"), traces.get("omnivoice_e2")
    if ob and oe:
        k = next(iter(ob.values()))["K"]
        same = sum(ob[i]["same_audio"] and oe[i]["same_audio"] for i in ob)
        rows.append(["OmniVoice, window frames, steps 1-%d" % k, len(ob), same,
                     pct(med([float(r["in_natural"][:k].mean()) for r in ob.values()])),
                     pct(med([float(r["out_natural"][:k].mean()) for r in ob.values()])),
                     pct(med([float(r["in_natural"][:k].mean()) for r in oe.values()])),
                     pct(med([float(r["in_excited"][:k].mean()) for r in oe.values()])),
                     "x%.1f" % med([float(r["in_excited"][:k].mean() / r["in_natural"][:k].mean())
                                    for r in oe.values()]),
                     f"{pct(med([float(r['in_natural'][k:].mean()) for r in oe.values()]))} / "
                     f"{pct(med([float(r['in_natural'][k:].mean()) for r in ob.values()]))}"])
    cb, ce = traces.get("cosyvoice_base"), traces.get("cosyvoice_e2")
    if cb and ce:
        same = sum(cb[i]["same_audio"] and ce[i]["same_audio"] for i in cb)
        nat_b = [float(per_layer("cosyvoice", r)[0].mean()) for r in cb.values() if per_layer("cosyvoice", r)[0] is not None]
        pairs = [per_layer("cosyvoice", r) for r in ce.values()]
        pairs = [(float(n.mean()), float(x.mean())) for n, x in pairs if n is not None]
        rows.append(["CosyVoice 2, steps at the tag", len(cb), same, pct(med(nat_b)), "-",
                     pct(med([n for n, _ in pairs])), pct(med([x for _, x in pairs])),
                     "x%.1f" % med([x / n for n, x in pairs if n > 0]), "-"])
    if rows:
        lines += ["Attention share on the tag's tokens (median over items; mean over heads and layers).", "",
                  table(["", "items", "traces = judged audio", "base, in window", "base, outside window",
                         "e2, natural", "e2, excited", "excited / natural", "after step K: e2 / base"], rows), ""]
    else:
        lines += [f"Attention shares skipped: no traces in {attn} (run bench.excite.attn_omnivoice and "
                  "attn_cosyvoice).", ""]
    return traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args()
    j = Judged(args.out, args.judge)
    lines = ["# Excitation evidence", "", "## 1. The tag gets little attention, and raising it voices the tag", ""]
    traces = attention_section(args.out, lines)

    rows = []
    for model in ("omnivoice", "cosyvoice"):
        if j.has(f"{model}_base") and j.has(f"{model}_e2"):
            fixed, broken, p = paired(j.hits(f"{model}_base"), j.hits(f"{model}_e2"))
            rows.append([model, len(j.preds(f"{model}_base")), f"{j.metric(f'{model}_base'):.3f}",
                         f"{j.metric(f'{model}_e2'):.3f}", f"{fixed} / {broken}", fmt_p(p)])
    lines += ["Excitation alone (e2, b = 2) against no guidance.", "",
              table(["model", "items", "F1, base", "F1, e2", "fixed / broken", "p"], rows), ""]

    def against_base_and_e2(names, extra):
        hb, he = j.hits("omnivoice_base"), j.hits("omnivoice_e2")
        out = []
        for name in names:
            system = f"omnivoice_{name}"
            if not j.has(system):
                continue
            h = j.hits(system)
            f, b, p = paired(hb, h)
            row = [name, *extra(name), f"{j.metric(system):.3f}", f"{f} / {b}, p {fmt_p(p)}"]
            if name != "e2":
                better, worse, q = paired(h, he)
                row.append(f"{better} / {worse}, p {fmt_p(q)}")
            else:
                row.append("-")
            out.append(row)
        return out

    if j.has("omnivoice_base") and j.has("omnivoice_e2"):
        what = {"e2": "tag's tokens", "e2_prev": "word before the tag", "e2_first": "sentence's first word",
                "base_s1": "no bias, seed 1"}
        lines += ["Placebo and reseed (OmniVoice): the same bias, window and steps on another word's tokens.", "",
                  table(["setting", "bias on", "F1", "fixed / broken vs base", "e2 better / worse"],
                        against_base_and_e2(list(what), lambda n: [what[n]])), ""]

        lines += ["## 2. The tag is settled when its frames are committed", "",
                  "OmniVoice's schedule: share of frames committed before each step.", "",
                  table(["step"] + [str(s + 1) for s in (8, 16, 24, 28, 30)],
                        [["committed"] + [f"{100 * committed(s):.0f}%" for s in (8, 16, 24, 28, 30)]]), ""]
        k = sb.CONFIG["omnivoice"]["steps"]

        def steps(name):
            lo, hi = (0, k) if name == "e2" else CONTROLS[name][3]
            return [f"{lo + 1}-{hi}", f"{100 * committed(lo):.0f}%", hi - lo]

        lines += ["Excitation moved to later steps.", "",
                  table(["setting", "excited steps", "committed at start", "steps excited", "F1",
                         "fixed / broken vs base", "e2 better / worse"],
                        against_base_and_e2(["e2", "e2_late", "e2_all", "e2_from24", "e2_from28"], steps)), ""]

    lines += ["## 3. A window, because one tag spans many frames", ""]
    tok_rows = []
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(os.environ.get("OMNIVOICE_CKPT", "k2-fsa/OmniVoice"))
        tok_rows = [[t, len(tok(" " + t, add_special_tokens=False).input_ids)]
                    for t in sb.CONFIG["omnivoice"]["tags"].values()]
    except Exception as e:  # noqa: BLE001 - optional part
        lines += [f"Tag token counts skipped ({type(e).__name__}).", ""]
    if tok_rows:
        lines += ["OmniVoice text tokens per tag.", "", table(["tag", "tokens"], tok_rows), ""]
    ob = (traces or {}).get("omnivoice_base")
    if ob:
        w = [r["window"][1] - r["window"][0] + 1 for r in ob.values()]
        t = [r["t_len"] for r in ob.values()]
        lines += [f"OmniVoice window: median {med(w):g} frames ({med(w) / FRAME_RATE:.1f} s), range {min(w)}-{max(w)}; "
                  f"utterance median {med(t):g} frames ({len(w)} items).", ""]
    win_path = args.out / "windows.jsonl"
    if win_path.exists():
        rows = sb.read_jsonl(win_path)
        share = {}
        for r in rows:
            share.setdefault(r["system"], []).append(r["in_window"])
        lines += ["CosyVoice 2: share of decoding steps with the window on (mean over clips).", "",
                  table(["setting", "clips", "window on"],
                        [[s, len(v), f"{100 * statistics.mean(v):.1f}%"] for s, v in sorted(share.items())
                         if not s.endswith("_base")]), ""]
    text = "\n".join(lines)
    (args.out / "excite_evidence.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
