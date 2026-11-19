"""Summary of LCG's attention mask on NVV tags, split by whether the judge heard the tag in the base clip.

OmniVoice: from <out>/attn/omnivoice_lcg_base.pt (bench.guidance.lcg_omnivoice), the raw mask (frames whose peak
text token in layers 8 and 12 is the tag's) at the last guided step (8) and the last step (32), and where it falls
against our window; and, once judged, presence guidance with the LCG mask (omnivoice_p2_lcg) against our window (p2).

CosyVoice 2: from <out>/attn/cosyvoice_base.pt (bench.excite.attn_cosyvoice), the steps whose read position is on
the tag's tokens: an LCG-style mask for the autoregressive model, read with our probe (layers 11 and 16, mean over
heads), the one whose window is used. The tag's token span per item needs CosyVoice's tokenizer once; it is cached in
<out>/cosyvoice_tag_spans.json.

    python -m bench.guidance.lcg_report --out out

Writes <out>/lcg_mask.md and prints it.
"""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from bench import superbench as sb
from bench.stats import Judged, fmt_p, paired, table


def cosyvoice_tag_spans(out):
    """{item id: [first, last + 1)} of the tag's tokens in the target text, in CosyVoice 2's tokenizer."""
    path = out / "cosyvoice_tag_spans.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import sys

    sys.path.insert(0, str(sb.env_path("COSYVOICE_REPO")))
    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    tok = get_qwen_tokenizer(str(Path(sb.env_path("COSYVOICE_MODEL_DIR")) / "CosyVoice-BlankEN"),
                             skip_special_tokens=True).tokenizer
    spans = {}
    for it in sb.items("cosyvoice"):
        enc = tok(it["text"], return_offsets_mapping=True)
        s = it["text"].find(it["tag"])
        idx = [i for i, (x, y) in enumerate(enc.offset_mapping) if y > x and y > s and x < s + len(it["tag"])]
        spans[it["id"]] = [idx[0], idx[-1] + 1]
    path.write_text(json.dumps(spans), encoding="utf-8")
    return spans


def groups(outcomes, ids):
    """Items the judge heard the tag in (hit) and the rest (missed: off, other or none)."""
    return {"hit": [i for i in ids if outcomes[i] == "hit"], "missed": [i for i in ids if outcomes[i] != "hit"]}


def share(xs):
    return f"{sum(xs)}/{len(xs)} ({100 * sum(xs) / len(xs):.0f}%)" if xs else "-"


def med(xs):
    return f"{median(xs):g}" if xs else "-"


def omnivoice(out, judged, lines):
    trace = torch.load(out / "attn" / "omnivoice_lcg_base.pt")
    assert all(r["same_audio"] for r in trace.values()), "trace is not of the judged base clips"
    g = groups(judged.outcomes("omnivoice_base"), list(trace))
    rows = []
    for name, ids in g.items():
        for step, label in ((7, "8 (last guided)"), (-1, "32 (last)")):
            raw = [trace[i]["raw"][step] for i in ids]
            win = [trace[i]["window"] for i in ids]
            n = [int(r.sum()) for r in raw]
            inside = [int((r & w).sum()) / int(r.sum()) for r, w in zip(raw, win) if r.any()]
            rows.append([name, len(ids), label, share([x == 0 for x in n]), med(n),
                         f"{100 * median(inside):.0f}%" if inside else "-"])
    lines += ["## OmniVoice: LCG's mask in the base clips", "",
              "Raw mask: frames whose peak text token (layers 8 and 12, max over heads) is one of the tag's. "
              "Inside window: the share of its frames within our window (word before the tag to word after it).", "",
              table(["base outcome", "items", "step", "empty mask", "median frames", "median inside window"], rows),
              ""]
    if judged.has("omnivoice_p2_lcg"):
        base = judged.hits("omnivoice_base")
        rows = []
        for system, label in (("omnivoice_p2", "presence, our window"), ("omnivoice_p2_lcg", "presence, LCG mask")):
            b, w, p = paired(base, judged.hits(system))
            rows.append([label, f"{judged.metric(system):.3f}", f"{b} / {w}", fmt_p(p)])
        b, w, p = paired(judged.hits("omnivoice_p2_lcg"), judged.hits("omnivoice_p2"))
        lines += ["## OmniVoice: presence guidance (l1 = 2, steps 1-8) with each localization", "",
                  f"No guidance: F1 {judged.metric('omnivoice_base'):.3f}.", "",
                  table(["localization", "F1", "fixed / broken vs no guidance", "p"], rows), "",
                  f"Our window against the LCG mask, per item: better {b}, worse {w}, p = {fmt_p(p)}.", ""]


def cosyvoice(out, judged, lines):
    trace = torch.load(out / "attn" / "cosyvoice_base.pt")
    spans = cosyvoice_tag_spans(out)
    g = groups(judged.outcomes("cosyvoice_base"), list(trace))
    rows = []
    for name, ids in g.items():
        on = [sum(spans[i][0] <= int(p) < spans[i][1] for p in trace[i]["pos"]) for i in ids]
        win = [sum(trace[i]["lo"] <= int(p) < trace[i]["hi"] for p in trace[i]["pos"]) for i in ids]
        rows.append([name, len(ids), share([x == 0 for x in on]), med(on), share([x == 0 for x in win]), med(win)])
    lines += ["## CosyVoice 2: read position on the tag in the base clips", "",
              "Steps whose read position (peak text token, layers 11 and 16) is one of the tag's tokens, against "
              "steps in our window (word before the tag to word after it, +-1 token).", "",
              table(["base outcome", "items", "never on tag", "median steps on tag", "never in window",
                     "median steps in window"], rows), ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out"))
    args = ap.parse_args()
    judged = Judged(args.out)
    lines = ["# LCG's attention mask on NVV tags", ""]
    if (args.out / "attn" / "omnivoice_lcg_base.pt").exists():
        omnivoice(args.out, judged, lines)
    if (args.out / "attn" / "cosyvoice_base.pt").exists():
        cosyvoice(args.out, judged, lines)
    text = "\n".join(lines)
    (args.out / "lcg_mask.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
