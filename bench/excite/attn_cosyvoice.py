"""Attention trace of excitation on CosyVoice 2: how much the current speech token attends to the tag's tokens.

Re-synthesizes items through run_cosyvoice.py's guided inference (same seed, voice and window) and records, at every
decoding step and in every layer of the LM's conditional stream, the attention mass (mean over heads, over all keys:
prompt, text and speech so far) on the tag's text tokens:

  natural   the model's own attention, without the excitation bias
  excited   the attention actually used (equal to natural when b = 0 or outside the window)

It also records the read position after every step, so the steps at the tag are known in every setting, including
b = 0. Single-stream settings only (base, e2). The re-synthesized audio is compared with the clip in --out.

    python -m bench.attn_cosyvoice --out out [--ids en_101 ...]

Writes <out>/attn/cosyvoice_<setting>.pt (dict id -> arrays).
"""
import argparse
from pathlib import Path

import torch

from bench import run_cosyvoice
from bench import superbench as sb


def trace_reader():
    from tagguide.cosyvoice import Reader

    class TraceReader(Reader):
        """Reader that also records, on each recorded call (one per layer of a cond-stream step), the attention
        mass on the tag's keys, natural and as used."""

        def __init__(self, qwen):
            super().__init__(qwen)
            self.rows = []

        def __call__(self, module, query, key, value, attention_mask, **kwargs):
            if self.on and query.shape[2] == 1:
                k = self.repeat_kv(key, module.num_key_value_groups).float()
                s = (query.float() @ k.transpose(-1, -2))[0, :, 0] * module.scaling  # [H, L]
                natural = s.softmax(-1)[:, self.tag].sum(-1).mean()
                excited = natural
                if self.excite:
                    s = s.clone()
                    s[:, self.tag] += self.bias
                    excited = s.softmax(-1)[:, self.tag].sum(-1).mean()
                self.rows.append(torch.stack([natural, excited]))
            return super().__call__(module, query, key, value, attention_mask, **kwargs)

    return TraceReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--settings", nargs="+", default=["base", "e2"], choices=["base", "e2"])
    ap.add_argument("--ids", nargs="*", default=None, help="items to trace (default: all)")
    ap.add_argument("--seed", type=int, default=sb.CONFIG["seed"])
    run_cosyvoice.add_args(ap)
    args = ap.parse_args()

    import soundfile as sf

    from tagguide.cosyvoice import read_position

    items = [it for it in sb.items("cosyvoice") if args.ids is None or it["id"] in args.ids]
    cosy = run_cosyvoice.Cosy(args, trace_reader())
    reader = cosy.reader
    n_layers = cosy.lm.llm.model.config.num_hidden_layers
    (args.out / "attn").mkdir(exist_ok=True)
    for name in args.settings:
        result = {}
        for k, it in enumerate(items):
            job = cosy.job(it, sb.SETTINGS[name])
            reader.rows, reads = [], []

            def wrap(inference):  # collect each cond step's read rows before the loop resets them
                def traced(*a, **kw):
                    for tok in inference(*a, **kw):
                        reads.append(torch.stack(reader.mass) if reader.mass else None)
                        yield tok
                return traced

            torch.manual_seed(args.seed)
            audio = cosy.synth(it, job, wrap)
            # rows[j]: forward j + 1 (the prefill, forward 0, has no single query); pos[j]: read position after
            # forward j, which sets the window for forward j + 1, so rows[j] is in the window iff lo <= pos[j] < hi.
            rows = torch.stack(reader.rows).view(-1, n_layers, 2)  # [steps, layers, 2]
            pos = [read_position(m, job["layers"]) if m is not None and m.shape[0] == n_layers else -1 for m in reads]
            saved = torch.tensor(sf.read(args.out / f"cosyvoice_{name}" / f"{it['id']}.wav")[0], dtype=torch.float32)
            n = min(len(audio), len(saved))
            same = bool(len(audio) == len(saved)
                        and torch.corrcoef(torch.stack([audio[:n].float(), saved[:n]]))[0, 1] > 0.99)
            result[it["id"]] = {"natural": rows[:, :, 0].cpu(), "excited": rows[:, :, 1].cpu(),
                                "pos": torch.tensor(pos), "lo": job["window"][0] - job["dilate"],
                                "hi": job["window"][1] + job["dilate"], "type": it["type"], "same_audio": same}
            print(f"[{name} {k + 1}/{len(items)}] {it['id']} steps={rows.shape[0]} same_audio={same}", flush=True)
        torch.save(result, args.out / "attn" / f"cosyvoice_{name}.pt")


if __name__ == "__main__":
    main()
