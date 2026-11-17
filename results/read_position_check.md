# Read position of CosyVoice 2's LM

Spearman correlation between the attention peak's position in the target text and the decoding step.

| item | text tokens | steps | mean of all layers | mean of layers [11, 16] |
|---|---|---|---|---|
| en_101 | 26 | 179 | 0.10 | 1.00 |
| en_601 | 34 | 210 | 0.41 | 1.00 |
| en_801 | 36 | 286 | 0.45 | 0.99 |
| en_1251 | 15 | 117 | 0.08 | 0.99 |

| layer | min over items | mean over items |
|---|---|---|
| 0 | -0.09 | 0.08 |
| 1 | -0.18 | 0.01 |
| 2 | nan | nan |
| 3 | -0.19 | -0.03 |
| 4 | -0.17 | -0.03 |
| 5 | -0.05 | 0.01 |
| 6 | -0.11 | -0.00 |
| 7 | -0.06 | nan |
| 8 | 0.00 | 0.10 |
| 9 | -0.17 | 0.24 |
| 10 | -0.05 | 0.01 |
| 11 | 0.99 | 0.99 |
| 12 | 0.75 | 0.87 |
| 13 | -0.03 | 0.08 |
| 14 | nan | nan |
| 15 | -0.06 | 0.09 |
| 16 | 0.99 | 0.99 |
| 17 | nan | nan |
| 18 | -0.22 | -0.08 |
| 19 | -0.12 | -0.00 |
| 20 | 0.08 | 0.19 |
| 21 | -0.05 | 0.16 |
| 22 | 0.70 | 0.79 |
| 23 | -0.00 | 0.03 |
