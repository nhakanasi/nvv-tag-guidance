# The models' own confusion matrix

## OmniVoice (150 items, 99 reproduce the base clip)

### Mean KL(tag || variant) in the window

| type | items | notag | laughter | sigh | surprise-ah | surprise-oh | dissatisfaction-hnn |
|---|---|---|---|---|---|---|---|
| laugh | 50 | 0.258 | - | 0.108 | 0.129 | 0.154 | 0.068 |
| sigh | 50 | 0.391 | 0.194 | - | 0.144 | 0.181 | 0.195 |
| ah | 16 | 0.418 | 0.199 | 0.197 | - | 0.075 | 0.196 |
| oh | 16 | 0.358 | 0.216 | 0.231 | 0.062 | - | 0.225 |
| hmm | 18 | 0.396 | 0.179 | 0.262 | 0.238 | 0.250 | - |

### Confuser

| type | least separable tag | configured confuser | KL to it | KL to notag |
|---|---|---|---|---|
| laugh | dissatisfaction-hnn | sigh | 0.068 | 0.258 |
| sigh | surprise-ah | laughter | 0.144 | 0.391 |
| ah | surprise-oh | surprise-oh | 0.075 | 0.418 |
| oh | surprise-ah | surprise-ah | 0.062 | 0.358 |
| hmm | laughter | sigh | 0.179 | 0.396 |

### Locality of the tag's effect (notag KL)

| distance from window, frames (25/s) | points | notag KL | share of in-window |
|---|---|---|---|
| -200 to -76 | 3995 | 0.018 | 0.05 |
| -75 to -51 | 1882 | 0.023 | 0.07 |
| -50 to -26 | 2519 | 0.035 | 0.10 |
| -25 to -1 | 2940 | 0.069 | 0.21 |
| in window | 4691 | 0.334 | 1.00 |
| 1 to 25 | 3577 | 0.107 | 0.32 |
| 26 to 50 | 2956 | 0.044 | 0.13 |
| 51 to 75 | 2094 | 0.028 | 0.08 |
| 76 to 200 | 1880 | 0.021 | 0.06 |

### Presence KL and the base outcome

| base outcome | items (same audio) | median notag KL in window |
|---|---|---|
| hit | 90 | 0.350 |
| missed | 9 | 0.199 |

Mann-Whitney p = 0.0643

## CosyVoice 2 (8 items, 8 reproduce the base clip)

### Mean KL(tag || variant) in the window

| type | items | notag | [breath] | [quick_breath] | [laughter] | [cough] | [clucking] | [lipsmack] | [hissing] | [sigh] |
|---|---|---|---|---|---|---|---|---|---|---|
| breath | 1 | 0.040 | - | 0.007 | 0.084 | 0.180 | 0.116 | 0.055 | 0.114 | 0.010 |
| laugh | 1 | 0.060 | 0.040 | 0.048 | - | 0.072 | 0.063 | 0.048 | 0.042 | 0.049 |
| cough | 1 | 0.347 | 0.210 | 0.217 | 0.190 | - | 0.039 | 0.173 | 0.205 | 0.224 |
| clucking | 1 | 0.622 | 0.563 | 0.591 | 0.291 | 0.205 | - | 0.456 | 0.201 | 0.552 |
| quick_breath | 1 | 0.016 | 0.005 | - | 0.045 | 0.126 | 0.077 | 0.030 | 0.041 | 0.008 |
| hiss | 1 | 0.267 | 0.161 | 0.184 | 0.125 | 0.221 | 0.111 | 0.162 | - | 0.163 |
| sigh | 1 | 0.020 | 0.014 | 0.012 | 0.040 | 0.065 | 0.052 | 0.016 | 0.160 | - |
| lipsmack | 1 | 0.049 | 0.032 | 0.020 | 0.073 | 0.067 | 0.056 | - | 0.069 | 0.014 |

### Confuser

| type | least separable tag | configured confuser | KL to it | KL to notag |
|---|---|---|---|---|
| breath | [quick_breath] | [quick_breath] | 0.007 | 0.040 |
| laugh | [breath] | [breath] | 0.040 | 0.060 |
| cough | [clucking] | [breath] | 0.039 | 0.347 |
| clucking | [hissing] | [lipsmack] | 0.201 | 0.622 |
| quick_breath | [breath] | [breath] | 0.005 | 0.016 |
| hiss | [clucking] | [sigh] | 0.111 | 0.267 |
| sigh | [quick_breath] | [breath] | 0.012 | 0.020 |
| lipsmack | [sigh] | [clucking] | 0.014 | 0.049 |

### Locality of the tag's effect (notag KL)

| distance from window, read position, text tokens | points | notag KL | share of in-window |
|---|---|---|---|
| -50 to -9 | 357 | 0.006 | 0.04 |
| -8 to -5 | 126 | 0.007 | 0.05 |
| -4 to -1 | 166 | 0.007 | 0.05 |
| in window | 363 | 0.154 | 1.00 |
| 1 to 4 | 246 | 0.032 | 0.20 |
| 5 to 8 | 147 | 0.021 | 0.14 |
| 9 to 50 | 255 | 0.009 | 0.06 |

### Presence KL and the base outcome

| base outcome | items (same audio) | median notag KL in window |
|---|---|---|
| hit | 0 | - |
| missed | 8 | 0.054 |

Mann-Whitney p = nan
