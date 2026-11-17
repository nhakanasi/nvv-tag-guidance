"""Score synthesized clips with NVV-SuperBench's own NVV verifier and metrics.

For every system folder <out>/<system>/ (wavs named <id>.wav), runs the benchmark's predict_nvc.py (GT-conditioned
judge: is the target NVV present, and at which word) once per run, then its eval_metrics.py over all runs. The
reference set is the whole benchmark (2,250 items), so Coverage is the share of all items a system's tag inventory
covers (fixed by the inventory, unchanged by guidance), P / R / F1 are on the covered items with a 2-word collar,
CA-F1 is the harmonic mean of F1 and Coverage, NTD the normalized position error.

--model picks the judge: a Qwen-Omni model through qwen_judge.py (QWEN_API_KEY) or a Gemini model (GEMINI_API_KEY).
Numbers are comparable only under one judge. Predictions go to <out>/nvv_pred/<judge>/run<k>/<system>/, metrics to
<out>/nvv_pred/<judge>/<system>.metrics.json and a table to <out>/nvv_pred/<judge>/nvv_report.md. Re-running
resumes: the predictor skips clips it already judged, so a rerun on judged clips makes no API calls.

    python -m bench.score --out out [--systems omnivoice_base omnivoice_e2] [--runs 3]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bench import superbench as sb

NVV = sb.BENCH / "evaluation" / "objective_evaluation" / "nvv_precision_recall_f1_distance"
GT_EN = sb.BENCH / "dataset" / "nvbench_data_en.json"
KEYS = [("coverage", "Coverage"), ("precision_paired", "Precision"), ("recall_paired", "Recall"),
        ("f1_paired", "F1"), ("ca_f1", "CA-F1"), ("ntd_mean", "NTD")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems", nargs="*", default=None, help="system folders to score (default: all with wavs)")
    ap.add_argument("--runs", type=int, default=1, help="independent verifier runs (the benchmark paper uses 3)")
    ap.add_argument("--model", default=sb.CONFIG["judge"]["model"], help="judge: qwen*-omni-* or models/gemini-*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--delta", type=int, default=sb.CONFIG["judge"]["delta"])
    args = ap.parse_args()

    qwen = args.model.startswith("qwen")
    key = "QWEN_API_KEY" if qwen else "GEMINI_API_KEY"
    if not os.environ.get(key):
        sys.exit(f"{key} not set (environment or .env)")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    predictor = ["-m", "bench.qwen_judge"] if qwen else [str(NVV / "predict_nvc.py")]
    systems = args.systems or sorted(p.name for p in args.out.iterdir() if p.is_dir() and any(p.glob("*.wav")))
    pred_root = args.out / "nvv_pred" / args.model.split("/")[-1]  # one folder per judge model
    rows = []
    for system in systems:
        audio = (args.out / system).resolve()
        preds = []
        for k in range(1, args.runs + 1):
            out_dir = pred_root / f"run{k}" / system
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, *predictor, str(audio), "--gt_json", str(GT_EN),
                            "--output_dir", str(out_dir), "--system_name", system, "--model_name", args.model,
                            "--pos_unit", "word", "--max_workers", str(args.workers), "--upload_concurrency", "4",
                            "--input_mode", "bytes", "--fail_fast", "0"], env=env, check=True, cwd=sb.ROOT)
            preds += [str(p) for p in out_dir.glob("*.json") if not p.name.endswith((".raw.json", ".skipped.json"))]
        save = pred_root / f"{system}.metrics.json"
        # No --audio_dir: the reference set is the whole benchmark, so Coverage is its instance coverage.
        subprocess.run([sys.executable, str(NVV / "eval_metrics.py"), "--gt_json", str(GT_EN), "--pred_json", *preds,
                        "--delta", str(args.delta), "--pos_unit", "word",
                        "--per_tag", "1", "--save_json", str(save)], env=env, check=True)
        rows.append((system, json.loads(save.read_text(encoding="utf-8"))))
    report(pred_root, rows)


def report(out, rows):
    """One line per system: eval_metrics.py's numbers, mean over runs."""
    lines = ["| system | R | runs | " + " | ".join(h for _, h in KEYS) + " |", "|---" * (len(KEYS) + 3) + "|"]
    for system, m in rows:
        runs = m["runs"]
        cells = []
        for k, _ in KEYS:
            vals = [r[k] for r in runs if r.get(k) is not None]
            cells.append(f"{sum(vals) / len(vals):.4f}" if vals else "NA")
        lines.append(f"| {system} | {m['reference_size_R']} | {len(runs)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (out / "nvv_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
