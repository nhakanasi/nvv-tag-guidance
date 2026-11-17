"""OmniVoice (base) on NVV-SuperBench with and without tag-local guidance.

Writes <out>/omnivoice_<setting>/<id>.wav and one row per clip to <out>/timing.jsonl. Resumes a stopped run.
Every setting of an item gets the same seed and reference voice.

    python bench/run_omnivoice.py --out out [--settings base e2] [--per_type 2]
"""
import argparse
import os
import time
from pathlib import Path

import superbench as sb

ARGS = [
    (("--steps",), {"type": int, "default": sb.CONFIG["omnivoice"]["steps"]}),
    (("--dilate",), {"type": float, "default": sb.CONFIG["omnivoice"]["dilate_s"]}),
    (("--ckpt",), {"default": None, "help": "OmniVoice checkpoint (default: OMNIVOICE_CKPT, else k2-fsa/OmniVoice)"}),
]


def add_args(ap):
    for flags, kw in ARGS:
        ap.add_argument(*flags, **kw)


def load(args):
    """OmniVoice with the whole-string tokenization patch and a Guide on its own tags."""
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    from tagguide.omnivoice import Guide, patch_tokenization
    from tagguide.tags import TagSet

    patch_tokenization()
    ckpt = args.ckpt or os.environ.get("OMNIVOICE_CKPT", "k2-fsa/OmniVoice")
    model = OmniVoice.from_pretrained(ckpt, device_map="cuda", dtype=torch.float16)
    tags = TagSet({k.strip("[]"): v.strip("[]") for k, v in sb.CONFIG["omnivoice"]["confuser"].items()})
    return model, Guide(model, tags, args.steps, args.dilate)


class Voices:
    """Voice-clone prompts of the reference voices, made once."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def __call__(self, item_id):
        ref_wav, ref_text = sb.ref_for(item_id)
        if ref_wav not in self.cache:
            self.cache[ref_wav] = self.model.create_voice_clone_prompt(str(ref_wav), ref_text)
        return self.cache[ref_wav]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=list(sb.SETTINGS), choices=list(sb.SETTINGS))
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    ap.add_argument("--max_vram_gb", type=float, default=0, help="cap this process's GPU memory (0: no cap)")
    add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    if args.max_vram_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb * 2**30 / total), 0)
    items = sb.items("omnivoice", per_type=args.per_type)
    args.out.mkdir(parents=True, exist_ok=True)
    done = sb.done_rows(args.out)
    model, guide = load(args)
    voices = Voices(model)
    for k, it in enumerate(items):
        prompt = voices(it["id"])
        if k == 0:  # untimed warm-up of every shape
            for name in args.settings:
                guide.setting = sb.SETTINGS[name]
                model.generate(text=it["text"], language="en", voice_clone_prompt=prompt)
        for name in args.settings:
            system = f"omnivoice_{name}"
            if (system, it["id"]) in done:
                continue
            (args.out / system).mkdir(exist_ok=True)
            guide.setting = sb.SETTINGS[name]
            torch.manual_seed(args.seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            audio = model.generate(text=it["text"], language="en", voice_clone_prompt=prompt)[0]
            torch.cuda.synchronize()
            gen_s = time.perf_counter() - start
            audio = torch.as_tensor(audio).squeeze().float().cpu().numpy()
            sf.write(args.out / system / f"{it['id']}.wav", audio, model.sampling_rate)
            sb.append_jsonl(args.out / "timing.jsonl", {"id": it["id"], "system": system, "type": it["type"],
                                                       "gen_s": round(gen_s, 3),
                                                       "out_s": round(len(audio) / model.sampling_rate, 2)})
        print(f"[omnivoice {k + 1}/{len(items)}] {it['id']}", flush=True)


if __name__ == "__main__":
    main()
