# Excitation evidence

## 1. The tag gets little attention, and raising it voices the tag

Attention share on the tag's tokens (median over items; mean over heads and layers).

|  | items | traces = judged audio | base, in window | base, outside window | e2, natural | e2, excited | excited / natural | after step K: e2 / base |
|---|---|---|---|---|---|---|---|---|
| OmniVoice, window frames, steps 1-8 | 150 | 150 | 1.00% | 0.81% | 1.01% | 5.08% | x5.0 | 1.12% / 1.11% |
| CosyVoice 2, steps at the tag | 400 | 400 | 0.73% | - | 0.70% | 2.87% | x4.0 | - |

Excitation alone (e2, b = 2) against no guidance.

| model | items | F1, base | F1, e2 | fixed / broken | p |
|---|---|---|---|---|---|
| omnivoice | 150 | 0.909 | 0.970 | 12 / 1 | .003 |
| cosyvoice | 400 | 0.551 | 0.603 | 61 / 38 | .027 |

Placebo and reseed (OmniVoice): the same bias, window and steps on another word's tokens.

| setting | bias on | F1 | fixed / broken vs base | e2 better / worse |
|---|---|---|---|---|
| e2 | tag's tokens | 0.970 | 12 / 1, p .003 | - |
| e2_prev | word before the tag | 0.916 | 8 / 8, p 1 | 13 / 2, p .007 |
| e2_first | sentence's first word | 0.926 | 6 / 4, p .754 | 11 / 2, p .022 |
| base_s1 | no bias, seed 1 | 0.943 | 11 / 5, p .210 | 6 / 1, p .125 |

## 2. The tag is settled when its frames are committed

OmniVoice's schedule: share of frames committed before each step.

| step | 9 | 17 | 25 | 29 | 31 |
|---|---|---|---|---|---|
| committed | 3% | 9% | 23% | 41% | 60% |

Excitation moved to later steps.

| setting | excited steps | committed at start | steps excited | F1 | fixed / broken vs base | e2 better / worse |
|---|---|---|---|---|---|---|
| e2 | 1-8 | 0% | 8 | 0.970 | 12 / 1, p .003 | - |
| e2_late | 9-32 | 3% | 24 | 0.973 | 10 / 0, p .002 | 3 / 2, p 1 |
| e2_all | 1-32 | 0% | 32 | 0.970 | 13 / 2, p .007 | 1 / 1, p 1 |
| e2_from24 | 25-32 | 23% | 8 | 0.933 | 5 / 1, p .219 | 8 / 1, p .039 |
| e2_from28 | 29-32 | 41% | 4 | 0.919 | 3 / 3, p 1 | 12 / 1, p .003 |

## 3. A window, because one tag spans many frames

OmniVoice text tokens per tag.

| tag | tokens |
|---|---|
| [laughter] | 3 |
| [sigh] | 4 |
| [surprise-ah] | 6 |
| [surprise-oh] | 6 |
| [dissatisfaction-hnn] | 7 |

OmniVoice window: median 30 frames (1.2 s), range 16-124; utterance median 168 frames (150 items).

CosyVoice 2: share of decoding steps with the window on (mean over clips).

| setting | clips | window on |
|---|---|---|
| cosyvoice_e2 | 400 | 28.3% |
| cosyvoice_p2 | 400 | 27.3% |
| cosyvoice_p2t3e2 | 400 | 27.5% |
| cosyvoice_t3 | 400 | 25.3% |
