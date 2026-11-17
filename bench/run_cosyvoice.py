"""CosyVoice 2 (0.5B) on NVV-SuperBench with and without tag-local guidance.

Writes <out>/cosyvoice_<setting>/<id>.wav, <out>/timing.jsonl and <out>/windows.jsonl (share of decoding steps
whose read position was at the tag). Resumes a stopped run. Needs a CosyVoice checkout (COSYVOICE_REPO) and the
CosyVoice2-0.5B model (COSYVOICE_MODEL_DIR).

    python bench/run_cosyvoice.py --out out [--settings base e2] [--per_type 2]
"""
import argparse
import sys
import time
from pathlib import Path

import superbench as sb

ARGS = [
    (("--dilate",), {"type": int, "default": sb.CONFIG["cosyvoice"]["dilate"]}),
    (("--layers",), {"type": int, "nargs": "+", "default": sb.CONFIG["cosyvoice"]["layers"]}),
    (("--model_dir",), {"default": None, "help": "CosyVoice2-0.5B (default: COSYVOICE_MODEL_DIR)"}),
]


def add_args(ap):
    for flags, kw in ARGS:
        ap.add_argument(*flags, **kw)


class Cosy:
    """CosyVoice 2 with the guided LM inference swapped in per utterance."""

    def __init__(self, args, reader_cls=None):
        repo = Path(sb.env_path("COSYVOICE_REPO"))
        sys.path[:0] = [str(repo), str(repo / "third_party" / "Matcha-TTS")]
        from cosyvoice.cli.cosyvoice import CosyVoice2

        from tagguide.cosyvoice import Reader

        self.args = args
        self.cosy = CosyVoice2(args.model_dir or sb.env_path("COSYVOICE_MODEL_DIR"), load_jit=False, load_trt=False,
                               fp16=False)
        self.lm = self.cosy.model.llm
        self.reader = (reader_cls or Reader)(self.lm.llm.model)
        self.hf_tok = self.cosy.frontend.tokenizer.tokenizer
        self.original = self.lm.inference
        self.sample_rate = self.cosy.sample_rate

    def encode(self, text):
        ids, _ = self.cosy.frontend._extract_text_token(text)
        return ids

    def job(self, it, setting):
        from tagguide.cosyvoice import make_job

        return make_job(self.hf_tok, self.encode, it["text"], it["tag"], it["notag"], it["confuser"], setting,
                        self.args.dilate, self.args.layers)

    def synth(self, it, job, wrap=None):
        """Audio (1-D tensor) of item `it` under `job`; wrap(inference) may wrap the guided generator."""
        import torch

        from tagguide.cosyvoice import guided_inference

        ref_wav, ref_text = sb.ref_for(it["id"])
        inference = guided_inference(self.lm, self.reader, job)
        self.lm.inference = wrap(inference) if wrap else inference
        try:
            chunks = [o["tts_speech"] for o in self.cosy.inference_zero_shot(it["text"], ref_text, str(ref_wav),
                                                                               stream=False, text_frontend=False)]
        finally:
            self.lm.inference = self.original
        return torch.cat(chunks, dim=1).squeeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=list(sb.SETTINGS), choices=list(sb.SETTINGS))
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch

    items = sb.items("cosyvoice", per_type=args.per_type)
    args.out.mkdir(parents=True, exist_ok=True)
    done = sb.done_rows(args.out)
    cosy = Cosy(args)
    torch.manual_seed(args.seed)
    cosy.synth(items[0], cosy.job(items[0], sb.SETTINGS[args.settings[-1]]))  # untimed warm-up
    for k, it in enumerate(items):
        for name in args.settings:
            system = f"cosyvoice_{name}"
            if (system, it["id"]) in done:
                continue
            job = cosy.job(it, sb.SETTINGS[name])
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
