"""Controls for presence guidance on OmniVoice: when must it act, and does it need the window?

Same code path, seed, voice and l1 = 2 as p2 (contrast in steps 1-8, on our window), changing one thing:

  p2_all      contrast in all 32 steps, as LCG guides every denoising step
  p2_from8    contrast in steps 9-32 only (about 3% of frames committed when it starts)
  p2_from24   contrast in steps 25-32 only (about 23% committed)
  p2_global   steps 1-8, but on every frame instead of the window (m = 1)

If p2_all is no better than p2 and the late ranges lose the gain, the contrast has to act before the window is
committed. If p2_global is worse than p2, the window matters. Writes <out>/omnivoice_<setting>/<id>.wav and timing
rows; score them with `bench.score --systems omnivoice_p2_all ...`.

    python -m bench.guidance.control_omnivoice --out out [--settings p2_all p2_from8 p2_from24 p2_global]
"""
import argparse
import time
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb

# name -> (contrast steps [start, end); None: the first K, window: True for ours, False for every frame)
CONTROLS = {"p2_all": ((0, 32), True), "p2_from8": ((8, 32), True), "p2_from24": ((24, 32), True),
            "p2_global": (None, False)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=list(CONTROLS), choices=list(CONTROLS))
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    items = sb.items("omnivoice", per_type=args.per_type)
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    windowed = guide.step_window
    done = sb.done_rows(args.out)
    for name in args.settings:
        guide.contrast_steps, use_window = CONTROLS[name]
        guide.step_window = windowed if use_window else (lambda step, w: torch.ones_like(w))
        guide.setting = sb.SETTINGS["p2"]
        system = f"omnivoice_{name}"
        (args.out / system).mkdir(parents=True, exist_ok=True)
        for k, it in enumerate(items):
            if (system, it["id"]) in done:
                continue
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = model.generate(text=it["text"], language="en", voice_clone_prompt=voices(it["id"]))[0]
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - t0
            audio = torch.as_tensor(audio).squeeze().float().cpu().numpy()
            sf.write(args.out / system / f"{it['id']}.wav", audio, model.sampling_rate)
            sb.append_jsonl(args.out / "timing.jsonl", {"id": it["id"], "system": system, "type": it["type"],
                                                       "gen_s": round(gen_s, 3),
                                                       "out_s": round(len(audio) / model.sampling_rate, 2)})
            print(f"[{name} {k + 1}/{len(items)}] {it['id']}", flush=True)


if __name__ == "__main__":
    main()
