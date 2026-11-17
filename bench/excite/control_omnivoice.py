"""Controls for attention excitation on OmniVoice: is it the attention to the tag that helps?

Synthesizes the OmniVoice items with the same code path, voice and window as run_omnivoice.py, in settings that
leave the tag's attention alone:

  e2_prev   excitation b = 2 in the tag's window, but on the tokens of the word before the tag (after it when the
            tag opens the sentence)  (placebo)
  e2_first  excitation b = 2 in the tag's window, but on the tokens of the sentence's first word  (placebo)
  base_s1   no guidance, seed 1 instead of 0  (how many items flip by chance)
  e2_late   excitation b = 2 on the tag, but only in steps 8-31 instead of the first 8  (does the model commit
            to the tag early? Attend-and-Excite stops early because late steps no longer move objects)
  e2_all    excitation b = 2 on the tag in all 32 steps
  e2_from24 / e2_from28   excitation b = 2 on the tag only from step 24 / 28 (OmniVoice's schedule, t_shift 0.1,
            has committed about 3% of frames by step 8, 23% by step 24 and 41% by step 28)

If a placebo fixes about as many items as e2, the gain is not from attending to the tag. If e2_late fixes about as
many as e2, the first steps are not where the tag is decided. Writes
<out>/omnivoice_<setting>/<id>.wav and timing rows, so `score.py --systems omnivoice_e2_prev ...` scores them.

    python -m bench.control_omnivoice --out out [--settings e2_prev e2_first base_s1 e2_late e2_all]
"""
import argparse
import re
import time
from pathlib import Path

from bench import run_omnivoice
from bench import superbench as sb

# name -> (setting, placebo word, seed, excitation steps [start, end); None: the first K)
CONTROLS = {"e2_prev": ((0, 0, 2), "prev", 0, None), "e2_first": ((0, 0, 2), "first", 0, None),
            "base_s1": ((0, 0, 0), None, 1, None), "e2_late": ((0, 0, 2), None, 0, (8, 32)),
            "e2_all": ((0, 0, 2), None, 0, (0, 32)), "e2_from24": ((0, 0, 2), None, 0, (24, 32)),
            "e2_from28": ((0, 0, 2), None, 0, (28, 32))}
WORD = re.compile(r"\w[\w']*")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=list(CONTROLS), choices=list(CONTROLS))
    ap.add_argument("--per_type", type=int, default=0, help="first N items of each type (0: all)")
    run_omnivoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from omnivoice.models.omnivoice import _combine_text

    items = sb.items("omnivoice", per_type=args.per_type)
    model, guide = run_omnivoice.load(args)
    voices = run_omnivoice.Voices(model)
    tag_positions = guide.tag_token_positions
    state = {"placebo": None}

    def placebo_positions(task, text, cond_len, grouped=False):
        """The excitation keys: the tag's tokens, or for a placebo the tokens of another word of the target."""
        if state["placebo"] is None:
            return tag_positions(task, text, cond_len, grouped)
        prefix = "<|text_start|>"
        wrapped = f"{prefix}{_combine_text(ref_text=task.ref_texts[0], text=text)}<|text_end|>"
        start = len(prefix) + len(_combine_text(text=task.ref_texts[0])) + 1  # first character of the target
        tag = list(guide.tags.span.finditer(wrapped))[-1]
        before = list(WORD.finditer(wrapped, start, tag.start()))
        if state["placebo"] == "prev":  # the word before the tag, or after it when the tag opens the sentence
            word = before[-1] if before else WORD.search(wrapped, tag.end())
        else:  # the sentence's first word (not one inside the tag)
            word = before[0] if before else WORD.search(wrapped, tag.end())
        enc = model.text_tokenizer(wrapped, return_offsets_mapping=True)
        style_len = cond_len - len(enc.input_ids) - task.ref_audio_tokens[0].size(-1) - task.target_lens[0]
        group = [style_len + i for i, (a, b) in enumerate(enc.offset_mapping)
                 if b > a and a < word.end() and b > word.start()]
        state["words"].append(word.group(0))
        return [group] if grouped else group

    guide.tag_token_positions = placebo_positions
    done = sb.done_rows(args.out)
    for name in args.settings:
        setting, placebo, seed, guide.excite_steps = CONTROLS[name]
        system = f"omnivoice_{name}"
        (args.out / system).mkdir(parents=True, exist_ok=True)
        guide.setting, state["placebo"] = setting, placebo
        for k, it in enumerate(items):
            if (system, it["id"]) in done:
                continue
            state["words"] = []
            torch.manual_seed(seed)
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
            print(f"[{name} {k + 1}/{len(items)}] {it['id']} excited={state['words']}", flush=True)


if __name__ == "__main__":
    main()
