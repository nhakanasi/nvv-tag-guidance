"""Controls for presence guidance on CosyVoice 2: how must the window be placed in an autoregressive model?

Same code path, seed, voice and l1 = 2 as p2 (contrast while the read position is between the word before the tag
and the word after it, +-1 token), changing only where it acts:

  p2_tagpeak  LCG-style: on while the read position is on the tag's own tokens, then for HOLD more steps (an
              autoregressive model cannot widen the mask backwards: those tokens are already emitted)
  p2_global   at every step (m = 1)

Writes <out>/cosyvoice_<setting>/<id>.wav, timing and window rows; score with `bench.score`.

    python -m bench.guidance.control_cosyvoice --out out [--settings p2_tagpeak p2_global]
"""
import argparse
import time
from pathlib import Path

from bench import run_cosyvoice
from bench import superbench as sb

HOLD = 10  # steps (0.4 s at 25 tokens/s), as LCG's dilation radius of 10 frames
CONTROLS = ["p2_tagpeak", "p2_global"]


def tag_peak(tag):
    """Gate: on while the read position is on the tag's tokens and for HOLD steps after."""
    state = {"left": 0}

    def gate(pos):
        if pos in tag:
            state["left"] = HOLD
            return True
        if state["left"] > 0:
            state["left"] -= 1
            return True
        return False
    return gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=CONTROLS, choices=CONTROLS)
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_cosyvoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    items = sb.items("cosyvoice", per_type=args.per_type)
    done = sb.done_rows(args.out)
    cosy = run_cosyvoice.Cosy(args)
    for k, it in enumerate(items):
        for name in args.settings:
            system = f"cosyvoice_{name}"
            if (system, it["id"]) in done:
                continue
            job = cosy.job(it, sb.SETTINGS["p2"])
            job["gate"] = tag_peak(set(job["tag"])) if name == "p2_tagpeak" else (lambda pos: True)
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            audio = cosy.synth(it, job).numpy()
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - start
            (args.out / system).mkdir(exist_ok=True)
            sf.write(args.out / system / f"{it['id']}.wav", audio, cosy.sample_rate)
            sb.append_jsonl(args.out / "windows.jsonl",
                            {"id": it["id"], "system": system, "in_window": round(job.get("in_window", 0.0), 4)})
            sb.append_jsonl(args.out / "timing.jsonl", {"id": it["id"], "system": system, "type": it["type"],
                                                       "gen_s": round(gen_s, 3),
                                                       "out_s": round(len(audio) / cosy.sample_rate, 2)})
        print(f"[cosyvoice {k + 1}/{len(items)}] {it['id']}", flush=True)


if __name__ == "__main__":
    main()
