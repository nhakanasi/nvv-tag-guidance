# LCG's attention mask on NVV tags

## OmniVoice: LCG's mask in the base clips

Raw mask: frames whose peak text token (layers 8 and 12, max over heads) is one of the tag's. Inside window: the share of its frames within our window (word before the tag to word after it).

| base outcome | items | step | empty mask | median frames | median inside window |
|---|---|---|---|---|---|
| hit | 136 | 8 (last guided) | 0/136 (0%) | 15.5 | 100% |
| hit | 136 | 32 (last) | 0/136 (0%) | 17 | 88% |
| missed | 14 | 8 (last guided) | 0/14 (0%) | 15.5 | 100% |
| missed | 14 | 32 (last) | 0/14 (0%) | 11 | 88% |

## OmniVoice: presence guidance (l1 = 2, steps 1-8) with each localization

No guidance: F1 0.909.

| localization | F1 | fixed / broken vs no guidance | p |
|---|---|---|---|
| presence, our window | 0.967 | 12 / 2 | .013 |
| presence, LCG mask | 0.977 | 13 / 2 | .007 |

Our window against the LCG mask, per item: better 2, worse 3, p = 1.

## CosyVoice 2: read position on the tag in the base clips

Steps whose read position (peak text token, layers 11 and 16) is one of the tag's tokens, against steps in our window (word before the tag to word after it, +-1 token).

| base outcome | items | never on tag | median steps on tag | never in window | median steps in window |
|---|---|---|---|---|---|
| hit | 213 | 43/213 (20%) | 4 | 0/213 (0%) | 49 |
| missed | 187 | 55/187 (29%) | 3 | 0/187 (0%) | 41 |
