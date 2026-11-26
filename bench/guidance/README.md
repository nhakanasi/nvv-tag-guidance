# Tag-local presence guidance for NVV tags: motivation and evidence

Why we guide a TTS model towards the non-verbal tag it was given by contrasting it with itself on the same text
without the tag, only around the tag; and what in our models and data supports each motivation. Numbers are from
NVV-SuperBench (English, tag-based), judge `qwen3.8-omni-flash`, one run, seed 0, presence l1 = 2 (`p2`), unless
stated. WER and SIM-o are computed locally (Whisper large-v3; WavLM-large ECAPA-TDNN, OmniVoice's SIM-o model).

The method, on the model's log-probabilities:

    lp  +  l1 · m · (c − c_notag)

c is the conditional log-probability for the text with the tag, c_notag for the same text with the tag removed,
and m the tag's window (1 inside, 0 outside). As in classifier-free guidance, the model is contrasted with itself;
here the two inputs differ only by the tag, so the difference is what the tag asks for.

## Motivations

1. **Inference only: no retraining, no new data, low cost.** Tagged NVV speech is scarce and fine-tuning a TTS model
   for each tag inventory is slow. Guidance uses the model as it is, with one extra conditional pass.
2. **The model already has the capacity; it fails by not voicing the tag, or voicing it too weakly.** Missed tags
   are rarely the wrong sound. They are no sound, or a sound too faint or misplaced to be heard as the tag. At the
   tag's frames the tag removes what the model would otherwise put there, but spreads the probability over many
   tokens instead of choosing one: the model knows something different goes there, not what. Guidance concentrates
   that probability on the tokens the tag raised.
3. **A window keeps the fix on the tag.** The contrast acts only on the frames around the tag, so the rest of the
   utterance, including the speaker's voice, is the model's own.
4. **Like CFG, only where it matters, so no noticeable extra time.** On a masked-diffusion model (OmniVoice) it acts
   in the early steps, before the frames are committed; on an autoregressive model (CosyVoice 2) it acts while the
   model reads the tag.

## 1. Inference only

No weights change and no data is used. The cost is one extra conditional input per tag, the text without the tag:

| Model | Extra work | RTF, no guidance | RTF, presence |
|---|---|---|---|
| OmniVoice | one more row in the batch, first 8 of 32 steps | 0.153 | 0.153 |
| CosyVoice 2 | one more LM stream | 0.80 | 1.33 |

See section 4 for the CosyVoice 2 cost.

## 2. The model can voice the tag but under-voices it

**Misses are no sound or a misplaced sound, rarely a wrong one.** The verifier is told the target type, and its list
of NVVs heard but not asked for splits the outcomes. Hit: target within 2 words of the tag. Off: target heard
further away. Other: target not heard, another NVV heard. None: nothing heard.

| Model | Hit | Off | Other | None |
|---|---|---|---|---|
| OmniVoice, no guidance | 136 | 11 | 0 | 3 |
| CosyVoice 2, no guidance | 213 | 113 | 43 | 31 |

**The model knows where the tag is.** OmniVoice's own attention finds the tag. The frames whose peak text token in
layers 8 and 12 belongs to the tag (LCG's mask) are non-empty on every item, missed ones included: a median of 15.5
frames at step 8 for both hit and missed items, all inside our window (`results/lcg_mask.md`).

**It knows a sound goes there more than which sound.** On the unguided trajectory we compare the model's prediction
for the text with the tag against the same text with the tag removed, or with each other tag in its place. The
measure is the KL divergence in the tag's window, averaged over the first 8 steps and the codebooks (OmniVoice, 150
items, `results/confusion.md`):

| Asked for | items | KL to no tag | KL to the closest other tag |
|---|---|---|---|
| laugh | 50 | 0.258 | 0.068 |
| sigh | 50 | 0.391 | 0.144 |
| ah | 16 | 0.418 | 0.075 |
| oh | 16 | 0.358 | 0.062 |
| hmm | 18 | 0.396 | 0.179 |

Removing the tag moves the prediction 2-5 times more than swapping it for another tag. Whether a sound goes there is
what the tag mainly encodes.

**Missed tags move the prediction less.** The no-tag KL in the window has a median of 0.199 on the 9 missed items,
against 0.350 on the 90 hits (items whose trajectory reproduces the judged clip; Mann-Whitney p = .064, a trend).
The tag is there, but weakly: one token among many, trained at the same cost as any word.

**At its frames the tag removes the default but does not pick a sound.** One frame inside the window of a missed
item (en_1428, a laugh), at the first decoding step, codebook 1 (`results/figs/fig_frame.png`):

![The tag removes the default but does not pick a sound](results/figs/fig_frame.png)

| | top token | probability of the default (token 244) |
|---|---|---|
| without the tag | 244, 0.81 | 0.81 |
| with the tag | 774, 0.07 (flat) | 0.05 |
| presence guidance | 296, 0.60 | about 0 |

Without the tag the model is sure what goes there. With the tag it drops that, but the probability spreads over many
tokens the tag raised a little, and none of them stands out. Over the 89 items whose trace reproduces the base clip,
the top-token probability in the window at step 1 has a median of:

| | median | items, against "with the tag" | p (Wilcoxon) |
|---|---|---|---|
| window, without the tag | 0.101 | higher on 80 of 89 | 4e-14 |
| window, with the tag (base) | 0.060 | | |
| window, presence guidance | 0.107 | higher on 86 of 89 | 1e-15 |
| outside the window (base) | 0.088 | higher on 65 of 89 | 2e-6 |

The tag makes its own frames the least confident in the clip. OmniVoice commits the most confident frames first, so
the tag's frames are filled last, when the words around them are fixed and pull them back towards speech or silence:
the none and off outcomes. Guidance makes them more confident than the rest of the clip. (That the window's frames
are committed later, and earlier under guidance, is the expected consequence; it is not yet measured. The flatness
does not separate hits from misses, 0.060 against 0.066: it is the risk on every tagged item, and the misses are where
it resolves the wrong way.)

**Guidance concentrates the probability on what the tag raised.** In probabilities, `lp + l1 (c − c_notag)` samples
from

    p ∝ p_c · (p_c / p_notag)^l1

At l1 = 2:

- a token the tag made twice as likely (0.10 with the tag, 0.05 without) is multiplied by 4;
- a token the tag made less likely (the next word, 0.6 against 0.8) is multiplied by 0.56;
- a token the tag does not affect is left alone;
- the default in the frame above (0.81 without the tag, 0.05 with it) is multiplied by (0.05 / 0.81)^2 ≈ 0.004.

So the default is removed for good and the mass goes to the few tokens the tag raised most.

**It fixes misses on both models.** Fixed / broken counts the items missed without guidance and hit with it, and the
reverse; p is the exact two-sided sign test:

| Model | F1, no guidance | F1, presence | Fixed / broken | p |
|---|---|---|---|---|
| OmniVoice (150 items) | 0.909 | 0.967 | 12 / 2 | .013 |
| CosyVoice 2 (400 items) | 0.551 | 0.601 | 68 / 45 | .038 |

The outcomes move as intended:

- None falls: OmniVoice 3 → 0, CosyVoice 2 31 → 16.
- Off falls: OmniVoice 11 → 4, CosyVoice 2 113 → 85.
- Presence does not choose the sound: on CosyVoice 2, other rises (43 → 63).

What it sounds like, on the same seed (`results/figs/fig_example.png`): a missed item with no sound at the tag (en_1428)
and one with a faint sound (en_1414). With guidance a laugh appears inside the window and the speech outside it is
unchanged; FlexSED's laughter probability goes from 0 and 0.14 (under its 0.5 threshold) to 0.97. Over the 100
OmniVoice laugh and sigh items, FlexSED's peak in the window rises on 68 and falls on 23 (median 0.002 → 0.13,
p = 3e-6).

![Before and after presence guidance](results/figs/fig_example.png)

**It amplifies a tag the model reads; it cannot replace reading it.** CosyVoice 2's attention read position (the
peak text token in LM layers 11 and 16) lands on the tag's own tokens for a median of only 3-4 steps, and never in
98 of 400 base items. That happens more often on missed items (29% against 20% of hits, p = .036). Splitting the
items by whether the read position reached the tag (post hoc, from the tag-peak control run):

| CosyVoice 2 items | n | Fixed / broken | p |
|---|---|---|---|
| read position reaches the tag | 302 | 55 / 26 | .0017 |
| never reaches it | 98 | 13 / 19 | .38 |

Where the model reads the tag, c and c_notag differ and guidance strengthens the tag. Where it reads straight
through, they barely differ and the guided clip is only another sample.

## 3. The window keeps the fix on the tag and the speaker's voice

**The tag's effect spans about a second around it.** Here is the no-tag KL by distance from the window, as a share
of the KL inside it:

| OmniVoice, frames (25/s) | -50 to -26 | -25 to -1 | window | 1 to 25 | 26 to 50 | further |
|---|---|---|---|---|---|---|
| share of in-window KL | 0.10 | 0.21 | 1.00 | 0.32 | 0.13 | 0.05-0.08 |

| CosyVoice 2 (8 items), read position, text tokens | -8 to -1 | window | 1 to 4 | 5 to 8 | further |
|---|---|---|---|---|---|
| share of in-window KL | 0.05 | 1.00 | 0.20 | 0.14 | 0.04-0.06 |

The window runs from the start of the word before the tag to the end of the word after it, widened by 0.3 s on
OmniVoice and by ±1 token on CosyVoice 2. In CosyVoice 2 the tag changes nothing before the read position reaches
the word before it, so the window opens there.

**Without the window the voice drifts (OmniVoice, 150 items).** The `p2_global` setting applies the same contrast on
every frame. It keeps the hits: 92 of 92 judged items, against 91 for the window on the same items (1 / 0, p = 1).
But it changes the speaker:

| Setting | WER | SIM-o | SIM-o vs no guidance | p (Wilcoxon) | clips losing > 0.1 SIM-o |
|---|---|---|---|---|---|
| no guidance | 0.056 | 0.814 | | | |
| presence, window | 0.061 | 0.810 | -0.004 | .18 | 1 |
| presence, every frame | 0.061 | 0.777 | -0.037 | .001 | 12 |

Every frame against the window: SIM-o -0.033 (p = .013), WER equal (11 better / 9 worse). Outside the tag the
contrast is small but not zero, and summed over the clip it moves the voice in about one clip in twelve. The window
gives the same hits at the base model's voice. Against no guidance, the window raises WER on slightly more items
than it lowers it (21 / 9, p = .043), with no change in the median (0.056).

**Where the window comes from.** On OmniVoice it is placed with the model's own duration estimate of the text up to
each word boundary. LCG's attention mask works as well there (F1 0.977 against 0.967; 2 / 3 between them,
p = 1), so the window's exact form matters less than having one. On CosyVoice 2 it comes from the read position.

## 4. Only where it matters: early steps (NAR), the tag's steps (AR)

**OmniVoice: the first 8 of 32 steps, as early as CFG matters.** Its unmasking schedule commits 3% of frames by
step 8 and 23% by step 24, so the layout around the tag is open during the early steps. Guiding all 32 steps adds
nothing over the first 8: on the 146 items judged for both, 145 hits against 143 (2 / 0, p = .5). The extra row
runs in those 8 steps only, batched with the conditional and unconditional rows. The RTF is unchanged (0.153 with
and without guidance).

**CosyVoice 2: the steps at the tag.** An autoregressive model commits each speech token as it emits it, so the
contrast acts while the read position is in the window. That is a median of 25.5% of decoding steps, and at least
one step on every item. The contrast term is computed only then.

**Cost (to be measured).** The current implementation runs the no-tag stream at every step, sequentially, to keep
its KV cache in step with the emitted tokens. This is why its RTF is 1.33 against 0.80. Filling that cache in one
parallel prefill when the window opens, and running the stream only inside the window, would limit the extra decoding
to the ~25% of steps at the tag. That version is not yet run.

## Relation to LCG

Phrase-Localized Language-Contrastive Guidance (Lee et al., arXiv 2609.01016) also localizes a contrast in a speech
LM, for code-switched accent. It contrasts a phrase in one language with the same phrase in another, on frames from
an attention mask, at every step. Ours contrasts the tag with its absence. The alternative-style contrast (the tag
against another tag) is not significant here: 12 / 4, p = .077 on OmniVoice; 57 / 50, p = .56 on CosyVoice 2.
That fits section 2: the models rarely produce the wrong sound. Ours also acts in the early steps only on
OmniVoice, and is shown on an autoregressive model.

## Reproducing

From the judged predictions and traces in `results/` (no synthesis, no judge calls):

```bash
python -m bench.guidance.evidence --out results           # F1, fixed / broken, RTF, outcome split
python -m bench.guidance.lcg_report --out results         # LCG's mask (section 2, 3) -> results/lcg_mask.md
python -m bench.guidance.confusion_report --out results   # tag separability, locality (section 2, 3) -> results/confusion.md
python -m bench.guidance.figures --out results           # per-frame distributions (section 2) -> results/figs/fig_frame.png
```

`results/quality.jsonl` holds the per-clip WER and SIM-o (section 3, `results/quality_omnivoice_base.md`); the judged
predictions of the controls are under `results/nvv_pred/qwen3.8-omni-flash/run1/` (`omnivoice_p2_all`: 146 of 150
items judged, `omnivoice_p2_global`: 92).

To regenerate from scratch:

```bash
python -m bench.run_omnivoice --out out --settings base p2
python -m bench.run_cosyvoice --out out --settings base p2
python -m bench.guidance.control_omnivoice --out out --settings p2_all p2_global   # timing, window
python -m bench.guidance.control_cosyvoice --out out                               # tag-peak gate
python -m bench.guidance.lcg_omnivoice --out out                                   # LCG's mask
python -m bench.guidance.confusion_omnivoice --out out                             # tag separability
python -m bench.guidance.probe_omnivoice --out out --dump en_1428                  # per-frame distributions
python -m bench.score --out out                                                    # judge (Qwen API)
python -m bench.quality --out out --systems omnivoice_base omnivoice_p2 omnivoice_p2_global   # WER, SIM-o
```

`results/figs/fig_example.png` (spectrograms and FlexSED laughter probability) uses FlexSED from VocalSoundExtend, which this repo does not include.

## References

- Ho and Salimans, Classifier-Free Diffusion Guidance, 2022 ([arXiv 2207.12598](https://arxiv.org/abs/2207.12598))
- Lee et al., Phrase-Localized Language-Contrastive Guidance, EMNLP 2026 ([arXiv 2609.01016](https://arxiv.org/abs/2609.01016))
- NVV-SuperBench, 2026 ([arXiv 2604.16211](https://arxiv.org/abs/2604.16211))
