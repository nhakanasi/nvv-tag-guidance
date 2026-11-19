"""What is at the tag in the audio: no room, silence, a faint sound or a full one? Locally, no judge calls.

Whisper large-v3 (faster-whisper) word timestamps place the words around the tag; the tag's region runs from the end of the word
before it to the start of the word after it (words Whisper hears for the NVV itself, "haha", "sigh", fall inside).
For that region: its length, and its loudness relative to the clip's speech (frame RMS in dB, 25 ms frames, 10 ms
hop; speech level = median over the frames of the transcribed words): the 90th percentile frame (peak) and the
median. Items are grouped by the base outcome and by whether presence guidance fixed them.

    python -m bench.guidance.tag_energy --out out --model omnivoice [--systems base p2]

Appends to <out>/tag_energy.jsonl (resumes) and writes <out>/tag_energy_<model>.md.
"""
import argparse
import difflib
import statistics
from pathlib import Path

import numpy as np

from bench import superbench as sb
from bench.quality import norm
from bench.stats import Judged, table


def rms_db(x, sr):
    win, hop = int(0.025 * sr), int(0.010 * sr)
    n = max(1, 1 + (len(x) - win) // hop)
    frames = np.stack([x[i * hop:i * hop + win] for i in range(n)]) if len(x) >= win else x[None]
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-8), hop / sr


def region(words, ref, k):
    """(start, end) seconds of the tag region: after ref word k-1, before ref word k; None if unaligned."""
    hyp = [norm(w["text"]) for w in words]
    match = {}
    for blk in difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).get_matching_blocks():
        for j in range(blk.size):
            match[blk.a + j] = blk.b + j
    start = 0.0 if k == 0 else (words[match[k - 1]]["timestamp"][1] if k - 1 in match else None)
    end = None if k >= len(ref) else (words[match[k]]["timestamp"][0] if k in match else None)
    if start is None or (k < len(ref) and end is None):
        return None
    return start, end


def measure(asr, wav, text, tag):
    import soundfile as sf

    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    audio = audio.mean(1)
    before = text[:text.find(tag)]
    k = len(norm(before).split())
    ref = norm(text.replace(tag, " ")).split()
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr, 16000)
    x16 = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    segments, _ = asr.transcribe(x16, language="en", word_timestamps=True, beam_size=1)
    words = [{"text": w.word, "timestamp": (w.start, w.end)} for seg in segments for w in seg.words
             if norm(w.word)]
    span = region(words, ref, k)
    if span is None:
        return {"aligned": False}
    a, b = span
    b = len(audio) / sr if b is None else b
    db, step = rms_db(audio, sr)
    t = np.arange(len(db)) * step
    speech = np.zeros(len(db), bool)
    for w in words:
        s, e = w["timestamp"]
        speech |= (t >= s) & (t < (e if e is not None else t[-1] + 1))
    level = float(np.median(db[speech])) if speech.any() else float(np.median(db))
    gap = db[(t >= a) & (t < b)]
    return {"aligned": True, "dur": round(max(0.0, b - a), 3),
            "peak": round(float(np.percentile(gap, 90)) - level, 2) if len(gap) else None,
            "median": round(float(np.median(gap)) - level, 2) if len(gap) else None}


def summarize(out, model, systems):
    rows = {(r["system"], r["id"]): r for r in sb.read_jsonl(out / "tag_energy.jsonl")}
    j = Judged(out)
    base, guided = f"{model}_{systems[0]}", f"{model}_{systems[1]}"
    ob, og = j.outcomes(base), j.outcomes(guided)

    def stats(ids, system):
        rs = [rows[(system, i)] for i in ids if (system, i) in rows and rows[(system, i)]["aligned"]]
        if not rs:
            return [0, "-", "-", "-", "-"]
        dur = [r["dur"] for r in rs]
        peak = [r["peak"] for r in rs if r["peak"] is not None]
        return [len(rs), f"{statistics.median(dur):.2f}", f"{sum(d < 0.1 for d in dur) / len(dur):.0%}",
                f"{statistics.median(peak):.1f}" if peak else "-",
                f"{sum(p > -20 for p in peak) / len(peak):.0%}" if peak else "-"]

    groups = [("base hit", [i for i, o in ob.items() if o == "hit"])]
    for o in ("off", "other", "none"):
        groups.append((f"base {o}", [i for i, x in ob.items() if x == o]))
    fixed = [i for i in ob if ob[i] != "hit" and og.get(i) == "hit"]
    still = [i for i in ob if ob[i] != "hit" and og.get(i) != "hit"]
    head = ["items", "aligned", "region s (median)", "region < 0.1 s", "peak dB vs speech (median)",
            "peak > -20 dB"]
    lines = [f"## {model}", "", f"Base clips ({base}):", "",
             table(["group"] + head, [[g, len(ids)] + stats(ids, base) for g, ids in groups]), "",
             f"Misses fixed by guidance: the same items before ({base}) and after ({guided}):", "",
             table(["group"] + head, [["fixed, before", len(fixed)] + stats(fixed, base),
                                      ["fixed, after", len(fixed)] + stats(fixed, guided),
                                      ["not fixed, before", len(still)] + stats(still, base),
                                      ["not fixed, after", len(still)] + stats(still, guided)]), ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", required=True, choices=["omnivoice", "cosyvoice"])
    ap.add_argument("--systems", nargs=2, default=["base", "p2"])
    args = ap.parse_args()

    log = args.out / "tag_energy.jsonl"
    done = {(r["system"], r["id"]) for r in sb.read_jsonl(log)} if log.exists() else set()
    asr = None
    for name in args.systems:
        system = f"{args.model}_{name}"
        for k, it in enumerate(sb.items(args.model)):
            if (system, it["id"]) in done:
                continue
            if asr is None:
                from faster_whisper import WhisperModel

                asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
            row = measure(asr, args.out / system / f"{it['id']}.wav", it["text"], it["tag"])
            sb.append_jsonl(log, {"id": it["id"], "system": system, **row})
            if (k + 1) % 50 == 0:
                print(f"[{system} {k + 1}]", flush=True)
    text = summarize(args.out, args.model, args.systems)
    (args.out / f"tag_energy_{args.model}.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
