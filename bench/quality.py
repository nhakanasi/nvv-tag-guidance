"""Intelligibility and voice of the clips: WER and SIM-o per item, locally (no judge calls).

  WER    Whisper large-v3 transcript (English, greedy) against the item's text without the tag, with
         NVV-SuperBench's normalization (lowercase, punctuation removed except apostrophes); bracketed or
         parenthesized spans in the transcript ("[laughs]", "(sighs)") are dropped first.
  SIM-o  cosine similarity of the WavLM-large ECAPA-TDNN speaker embeddings of the clip and its reference voice
         (OmniVoice's own SIM-o model, k2-fsa/TTS_eval_models).

Runs in the OmniVoice environment (it has omnivoice.eval). Appends to <out>/quality.jsonl and resumes; prints the
mean per system and paired comparisons against the first system.

    python -m bench.quality --out out --systems omnivoice_base omnivoice_p2 omnivoice_p2_global
"""
import argparse
import re
import statistics
import string
from pathlib import Path

from bench import superbench as sb

PUNCT = "".join(c for c in string.punctuation if c != "'")
NVV_SPAN = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


def norm(text):
    text = NVV_SPAN.sub(" ", text).lower().translate(str.maketrans(PUNCT, " " * len(PUNCT)))
    return " ".join(text.split())


class Scorer:
    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        from omnivoice.eval.models.ecapa_tdnn_wavlm import ECAPA_TDNN_WAVLM
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.torch, self.dev = torch, "cuda"
        self.proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3",
                                                                   torch_dtype=torch.float16).to(self.dev).eval()
        root = Path(snapshot_download("k2-fsa/TTS_eval_models", allow_patterns=[
            "speaker_similarity/wavlm_large/*", "speaker_similarity/wavlm_large_finetune.pth"])) / "speaker_similarity"
        self.sv = ECAPA_TDNN_WAVLM(feat_dim=1024, channels=512, emb_dim=256, sr=16000,
                                   ssl_model_path=str(root / "wavlm_large") + "/")
        self.sv.load_state_dict(torch.load(root / "wavlm_large_finetune.pth", map_location="cpu")["model"],
                                strict=False)
        self.sv.to(self.dev).eval()
        self.ref_emb = {}

    def wav16(self, path):
        import soundfile as sf
        import torchaudio

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        x = self.torch.from_numpy(audio.mean(1))
        return torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x

    def transcribe(self, x):
        feats = self.proc(x.numpy(), sampling_rate=16000, return_tensors="pt").input_features
        with self.torch.no_grad():
            ids = self.asr.generate(feats.to(self.dev, self.torch.float16), language="en", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0]

    def embed(self, x):
        with self.torch.no_grad():
            return self.sv([x.to(self.dev)])

    def __call__(self, wav, ref_wav, text):
        import jiwer

        x = self.wav16(wav)
        hyp = self.transcribe(x)
        if ref_wav not in self.ref_emb:
            self.ref_emb[ref_wav] = self.embed(self.wav16(ref_wav))
        sim = float(self.torch.nn.functional.cosine_similarity(self.embed(x), self.ref_emb[ref_wav], dim=-1))
        return {"wer": jiwer.wer(norm(text), norm(hyp)), "sim": sim, "hyp": hyp}


WER_CAP = 1.0  # per clip: Whisper spells a long hiss as "s-s-s...", which gives WER above 10 on a few clips


def report(out, systems):
    from bench.stats import sign_test

    rows = [r for r in sb.read_jsonl(out / "quality.jsonl") if r["system"] in systems]
    by = {s: {r["id"]: r for r in rows if r["system"] == s} for s in systems}
    ids = sorted(set.intersection(*(set(v) for v in by.values())))
    ref = systems[0]
    lines = [f"| system | items | WER | SIM-o | WER lower / higher than {ref} | p | SIM-o higher / lower | p |",
             "|---|---|---|---|---|---|---|---|"]
    for s in systems:
        wer = statistics.fmean(min(by[s][i]["wer"], WER_CAP) for i in ids)
        sim = statistics.fmean(by[s][i]["sim"] for i in ids)
        cells = ["-"] * 4
        if s != ref:
            wl = sum(by[s][i]["wer"] < by[ref][i]["wer"] for i in ids)
            wh = sum(by[s][i]["wer"] > by[ref][i]["wer"] for i in ids)
            sh = sum(by[s][i]["sim"] > by[ref][i]["sim"] for i in ids)
            sl = sum(by[s][i]["sim"] < by[ref][i]["sim"] for i in ids)
            cells = [f"{wl} / {wh}", f"{sign_test(wl, wh):.3g}", f"{sh} / {sl}", f"{sign_test(sh, sl):.3g}"]
        lines.append(f"| {s} | {len(ids)} | {wer:.4f} | {sim:.4f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--systems", nargs="+", required=True, help="the first is the reference for the comparisons")
    args = ap.parse_args()

    log = args.out / "quality.jsonl"
    done = {(r["system"], r["id"]) for r in sb.read_jsonl(log)} if log.exists() else set()
    model = {s.split("_")[0] for s in args.systems}
    assert len(model) == 1, "one model per call (the items differ)"
    items = sb.items(model.pop())
    scorer = None
    for s in args.systems:
        for k, it in enumerate(items):
            if (s, it["id"]) in done:
                continue
            scorer = scorer or Scorer()
            ref_wav, _ = sb.ref_for(it["id"])
            row = scorer(args.out / s / f"{it['id']}.wav", str(ref_wav), it["notag"])
            sb.append_jsonl(log, {"id": it["id"], "system": s, **row})
            if (k + 1) % 25 == 0:
                print(f"[{s} {k + 1}/{len(items)}]", flush=True)
    text = report(args.out, args.systems)
    (args.out / f"quality_{args.systems[0]}.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
