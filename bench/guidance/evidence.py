"""Numbers behind bench/guidance/README.md, recomputed from judged predictions (no synthesis, no judge calls).

  1. Benchmark table: Coverage, P, R, F1, CA-F1, NTD, fixed / broken against no guidance with the sign test, RTF,
     for every setting of each model.
  2. Outcome split: hit / off / other / none per setting, and clips with an NVV that was not asked for.

    python -m bench.guidance.evidence --out results      # or --out out after your own runs

Writes <out>/guidance_evidence.md and prints it.
"""
import argparse
from pathlib import Path

from bench import superbench as sb
from bench.stats import Judged, fmt_p, paired, table

METRICS = [("coverage", "Coverage"), ("precision_paired", "P"), ("recall_paired", "R"), ("f1_paired", "F1"),
           ("ca_f1", "CA-F1"), ("ntd_mean", "NTD")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judge", default=None)
    args = ap.parse_args()
    j = Judged(args.out, args.judge)
    lines = ["# Guidance evidence", ""]
    bench_rows, outcome_rows = [], []
    for model in ("omnivoice", "cosyvoice"):
        base = f"{model}_base"
        if not j.has(base):
            continue
        hb = j.hits(base)
        for name, setting in sb.SETTINGS.items():
            system = f"{model}_{name}"
            if not j.has(system):
                continue
            cells = [f"{j.metric(system, k):.4f}" for k, _ in METRICS]
            if name == "base":
                fb = "-"
            else:
                fixed, broken, p = paired(hb, j.hits(system))
                fb = f"{fixed} / {broken}, p {fmt_p(p)}"
            bench_rows.append([model, name, ", ".join(map(str, setting)), *cells, fb, f"{j.rtf(system):.3f}"])
            o = list(j.outcomes(system).values())
            outcome_rows.append([model, name, len(o), *(o.count(k) for k in ("hit", "off", "other", "none")),
                                 j.extra_nvv(system)])
    lines += ["## 1. Benchmark", "", table(["model", "setting", "l1, l2, b", *(h for _, h in METRICS),
                                           "fixed / broken vs base", "RTF"], bench_rows), "",
              "## 2. Outcomes", "", table(["model", "setting", "items", "hit", "off", "other", "none",
                                          "clips with an NVV not asked for"], outcome_rows), ""]
    text = "\n".join(lines)
    (args.out / "guidance_evidence.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
