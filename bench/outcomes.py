"""Split each judged item into one outcome, from the verifier's existing predictions (no judge calls).

NVV-SuperBench's verifier is told the target type and only answers whether it is present and where, so a wrong
sound and no sound both end up as a miss. Its list of extra NVVs heard but not asked for ("hallucinated_events")
separates them, roughly:

  hit       target heard within --delta words of the marked position
  off       target heard, further away
  other     target not heard, but some other NVV heard somewhere in the clip (a proxy for confusion: the
            verifier gives only a loose location hint, so the other sound need not be at the tag)
  none      target not heard and nothing else either

    python bench/outcomes.py --out out [--judge qwen3.8-omni-flash] [--run 1]

Writes <out>/nvv_pred/<judge>/outcomes.md and prints it.
"""
import argparse
import collections
import json
from pathlib import Path


def outcome(p, delta):
    target = p["target_tag"].strip("<>")
    if p["present"]:
        pred, gt = p.get("pred_pos"), p.get("gt_pos")
        return "hit" if pred and gt and abs(pred["index"] - gt["index"]) <= delta else "off"
    others = [e.get("tag", "").strip("<>") for e in p.get("hallucinated_events") or []]
    return "other" if any(t and t != target for t in others) else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default="qwen3.8-omni-flash")
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--delta", type=int, default=2)
    args = ap.parse_args()

    root = args.out / "nvv_pred" / args.judge
    lines = ["| system | items | hit | off | other | none | other NVVs heard (top 3) |", "|---|---|---|---|---|---|---|"]
    for sys_dir in sorted((root / f"run{args.run}").iterdir()):
        preds = json.loads((sys_dir / f"{sys_dir.name}.json").read_text(encoding="utf-8")).values()
        counts, heard = collections.Counter(), collections.Counter()
        for p in preds:
            o = outcome(p, args.delta)
            counts[o] += 1
            if o == "other":
                heard.update(e.get("tag", "").strip("<>") for e in p["hallucinated_events"])
        top = ", ".join(f"{t} {n}" for t, n in heard.most_common(3))
        lines.append(f"| {sys_dir.name} | {sum(counts.values())} | " +
                     " | ".join(str(counts[k]) for k in ("hit", "off", "other", "none")) + f" | {top} |")
    text = "\n".join(lines) + "\n"
    (root / "outcomes.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
