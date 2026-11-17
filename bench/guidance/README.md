# Tag-local guidance: presence and type contrasts

Two classifier-free-guidance-style terms that act only in a window around each tag, on the model's
log-probabilities (`src/tagguide/contrast.py`):

    presence  + l1 * m * (c - c_notag)    c_notag: the same text with the tag removed
    type      + l2 * m * (c - c_conf)     c_conf: the tag replaced by a confusable tag

Numbers are from NVV-SuperBench (English, tag-based), judge `qwen3.8-omni-flash`, one run, seed 0.

## Motivation

A TTS model asked for an inline tag makes three kinds of error: the tag is **missed** (no sound), **confused** (the
wrong sound), or given **no room** (the words run through it). Each contrast compares the model with itself on a
text that differs only at the tag, so the difference isolates what the tag adds:

- **Presence.** Where the passes with and without the tag agree, the frames would sound the same either way and
  `c - c_notag` is zero; where they differ, the difference is what the tag asks for. Amplifying it pushes the window
  towards "something happens here" and away from reading through. It targets missed tags; it says nothing about
  which sound.
- **Type.** `c - c_conf` keeps only what separates the asked-for sound from its rival, so it targets confusions. It
  can only amplify a difference the model has learned; the confuser is ideally the tag the model most often
  produces instead, and here, without confusion counts for these base models, the acoustically closest other tag
  of the same inventory (`configs/superbench.yaml`).
- **Local.** Both terms are multiplied by the window m (word before the tag to word after it), so the rest of the
  utterance keeps the model's own scores. OmniVoice applies them in the first 8 of 32 steps, before its frames are
  committed (see `bench/excite/README.md`, section 2), alongside its own CFG; CosyVoice 2 applies them while its
  attention read position is at the tag, with no text CFG of its own (g = 0).

Each contrast costs one more conditional pass (OmniVoice, batched) or LM stream (CosyVoice 2, sequential).

## Evidence

**The contrasts fix misses on both models** (fixed / broken: items missed without guidance and hit with it, and
the reverse; p: exact two-sided sign test):

| Model | Setting | F1 | Fixed / broken | p | RTF |
|---|---|---|---|---|---|
| OmniVoice (150 items) | no guidance | 0.909 | | | 0.153 |
| | presence (l1 = 2) | 0.967 | 12 / 2 | .013 | 0.153 |
| | type (l2 = 3) | 0.966 | 12 / 4 | .077 | 0.152 |
| | presence + type + excitation | 0.970 | 13 / 2 | .007 | 0.158 |
| CosyVoice 2 (400 items) | no guidance | 0.551 | | | 0.80 |
| | presence | 0.601 | 68 / 45 | .038 | 1.33 |
| | type | 0.567 | 57 / 50 | .56 | 1.33 |
| | presence + type + excitation | 0.657 | 91 / 44 | <.001 | 1.86 |

**Where the gain comes from (outcome split).** The verifier is told the target type, so the benchmark scores a
wrong sound and no sound alike. Its list of NVVs heard but not asked for separates them roughly: hit (target within
2 words), off (target heard further away), other (target not heard, another NVV heard; a proxy for confusion),
none (nothing heard).

| Model | Setting | Hit | Off | Other | None |
|---|---|---|---|---|---|
| OmniVoice | no guidance | 136 | 11 | 0 | 3 |
| | presence | 146 | 4 | 0 | 0 |
| | type | 144 | 4 | 0 | 2 |
| CosyVoice 2 | no guidance | 213 | 113 | 43 | 31 |
| | presence | 236 | 85 | 63 | 16 |
| | type | 220 | 100 | 48 | 32 |
| | presence + type + excitation | 260 | 80 | 38 | 22 |

Most misses are placement (off), and guidance moves sounds onto the tag: presence turns off and none into hits on
both models. The combination also lowers "other" on CosyVoice 2 (43 to 38).

## Reproducing

From the judged predictions in `results/` (no synthesis, no judge calls):

```bash
python -m bench.guidance.evidence --out results    # -> results/guidance_evidence.md
```

To regenerate them: `python -m bench.run_omnivoice --out out`, `python -m bench.run_cosyvoice --out out`,
`python -m bench.score --out out`, then `python -m bench.guidance.evidence --out out`.
