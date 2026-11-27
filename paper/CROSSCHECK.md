# Method vs code cross-check

Each statement of the Method sections of `guidance.tex` (presence guidance) and `excite.tex` (attention excitation),
checked against the code in the `nvv-tag-guidance` repo (`src/tagguide/`, commit 3a2bc9a) and OmniVoice's upstream
code. Status: **ok** (code does what the paper says), **fixed** (paper wording changed to match the code),
**open** (a real difference, to decide).

## Open

1. **OmniVoice window: the duration estimate is not linear for short prefixes.** *Decision (option 1): described in both papers' OmniVoice window paragraph; a linear-window rerun is left for when judge calls are available.* The paper says the window boundary
   is the duration estimate of the untagged text up to a word boundary as a fraction of the estimate for the whole
   text. The code calls `duration_estimator.estimate_duration(prefix, ...)` (`src/tagguide/omnivoice.py:71-74`).
   Upstream (`omnivoice/utils/duration.py:208-249`) boosts estimates under `low_threshold = 50` frames (2 s) with a
   power curve, `50 * (d / 50) ** (1/3)`. A prefix whose linear estimate is short is therefore placed later than its
   share of the text. Measured on the 150 OmniVoice items (recomputed windows match the stored ones on 146), the boost
   changes the window on **62 items**, moving it later by up to about 20 frames (0.8 s). For example, en_16 gets
   frames 23-50 instead of 2-33. All reported results include this. The options are:
   - say so in the paper ("OmniVoice's duration estimator, including its boost for short texts");
   - or pass `low_threshold=None` in `Guide.window` and rerun every guided OmniVoice setting (no guidance does not
     use the window, so its clips stay).
   The window still covered the tag's sound well enough for the gains reported, but a linear window may do better on
   tags early in the sentence.

2. **CosyVoice 2 read position at excited steps.** The paper says the read position uses the attention "before the
   bias". The code computes each layer's scores before that layer's bias (`src/tagguide/cosyvoice.py:48-51`), but
   at an excited step the lower layers' bias has already changed the hidden states that layer 11 and 16 see. Strictly
   it is the pre-bias score of an excited forward pass. Minor; a footnote is enough.

## Fixed in the papers

3. **OmniVoice commits tokens, not frames.** The scheduler ranks and commits individual (frame, codebook) tokens
   (`omnivoice.py:139-144, 175-181`: `topk` over the flattened `[C, T]` scores, `total = t_len * C`). The papers said
   "commits frames" and "3% of frames committed"; now "tokens" and "3% of the tokens". The percentages themselves
   are of tokens and are unchanged.

## Checked, ok

| Statement in the paper | Code |
|---|---|
| OmniVoice CFG `l0 = c + g (c - u)`, g = 2 | `omnivoice.py:166-168` (`gen_config.guidance_scale`, default 2.0) |
| Presence `l = l0 + lambda m (c - c_notag)`, then log-softmax | `contrast.py:24`, `omnivoice.py:170-172` |
| `c_notag`: same text with the tag (and the space before it) removed | `tags.py:11, 22-24`; `contrast.texts` order `omnivoice.py:109` |
| `c_notag` computed on the same output so far | `omnivoice.py:183-185` (every row gets the committed tokens) |
| Each committed token is the arg max; confidence = top log-prob - codebook penalty + Gumbel noise | `omnivoice.py:173-179`; upstream default `class_temperature = 0` (greedy) |
| Terms act in the first K = 8 steps; later steps decode from l0 | `omnivoice.py:148-171` (`early = 0 <= s < 8`, 0-based) |
| Extra pass is one more batch row, first K steps only | `omnivoice.py:112-127, 153-159` |
| Window: word before the tag to word after it, +-0.3 s | `tags.py:30-38`, `omnivoice.py:76-85` (`dil = 0.3 * 25 = 7.5` frames) |
| Several tags: window is the union | `omnivoice.py:80-85` |
| Excitation: +b on pre-softmax scores, window frames to tag tokens, every layer and head, conditional row only | `excite.py:19-24` (additive float mask, row 0), `omnivoice.py:130-134, 152` (same mask in every layer) |
| Excitation adds no pass | `contrast.texts` returns only the text when l1 = l2 = 0 (`contrast.py:12-15`) |
| Odds multiplied by e^b (Eq. odds) | `excite.py:13-16` (`odds_boost`), exact for the tag set as a whole |
| Tag tokens: tokens overlapping the tag's characters, in the target text only | `omnivoice.py:93-103` |
| Whole-text tokenization in every setting | `omnivoice.py:31-35` (`patch_tokenization`) |
| CosyVoice 2: LM has no unconditional pass, l0 = c, CosyVoice's own sampling | `cosyvoice.py:133-136` (`sampling_ids`, top-k 25) |
| CosyVoice 2: `c_notag` is a second LM stream fed the same sampled tokens | `cosyvoice.py:113, 143-146` |
| Read position: attention of the current token over the target text (softmax over the text keys only), mean over heads, layers 11 and 16 (0-based), smoothed over 3 tokens, arg max | `cosyvoice.py:48-51, 87-90`; config `layers: [11, 16]` |
| Window on while the read position is in [word before, word after] +-1 token; terms act at the next step | `cosyvoice.py:117, 137-140` (`in_win` set after the step, used at the next) |
| Excitation on CosyVoice 2: +b from the current speech token to the tag's tokens, every layer and head, cond stream only, while the window is on | `cosyvoice.py:52-54, 125`; `excite.py:27-36` |
| Share of steps in the window (25-28%) | `cosyvoice.py:147`, logged to `windows.jsonl` |
| Probability form `p ∝ p0 (p_c / p_notag)^lambda` | exact: `exp` of Eq. presence after the log-softmax; the mask token is excluded afterwards (`omnivoice.py:173`) |
| fig_frame "without the tag" = `l0 + (1+g)(c_notag - c)` | `bench/guidance/probe_omnivoice.py:55`: equals `c_notag + g (c_notag - u)` with the same `u` |
| fig_attention "model's own attention" = scores without the bias | `bench/excite/probe_attn_omnivoice.py` (`mask.clamp(max=0)`; boolean mask at unexcited steps) |

## Minor, no change needed

- Read-position smoothing pads with zeros (`avg_pool1d`, `count_include_pad=True`), so the first and last text tokens
  are slightly down-weighted.
- `make_job` excites the first occurrence of the tag only (`cosyvoice.py:72`). The benchmark has one tag per item.
- CosyVoice 2 renormalizes `c` with a log-softmax inside the window even with no contrast (excitation only), which
  changes nothing.

## Final check (equations of Sections 3-4, reported numbers)

| Statement in the paper | Code / data | Status |
|---|---|---|
| Eq. window: `m_i = [A(j_s) - delta <= i < A(j_e) + delta]`, `j_s`/`j_e` start of the word before / end of the word after | `omnivoice.py` `window()`: `a = int(frames(plain[:start]) - dil)`, `b = int(frames(plain[:end]) + dil) + 1`; `tags.word_span` | ok (integer rounding of the edges) |
| OmniVoice `A(j) = T D(x_<j) / D(x)`, `x` the untagged text, `x_<j` its first `j` characters, `delta = 0.3 s` | `frames(prefix) = estimate_duration(prefix) * t_len / total`, `total = estimate_duration(plain)`; `dil = 0.3 * 25` | ok, boost described (open item 1) |
| CosyVoice 2 read position `r_t = argmax (kappa * a_t)`, `a_t` = softmax over the target-text keys, mean over heads, mean over layers {11, 16}, 3-token box filter | `Reader.__call__` (`softmax(-1).mean(0)` per layer), `read_position` (`mass[layers].mean(0)`, `avg_pool1d(3, 1, 1)`) | ok |
| `m_t = [j_s - delta <= r_{t-1} < j_e + delta]`, `delta = 1` token, the term at step `t` uses `r_{t-1}` | `lo <= pos < hi` with `lo, hi = window[0] - dilate, window[1] + dilate`; `in_win` set after the step and used at the next | ok |
| Gate `g_t = [t <= K]`, `K = 8` (OmniVoice) | `early = 0 <= s < 8` (0-based) | ok |
| Gate `g_t = m_t`, `O_t = {t}` (CosyVoice 2) | `in_win` gates `contrast.apply` and `reader.excite` | ok |
| Guidance 3.3: the `c_notag` pass "can be skipped" when `g_t = 0` | OmniVoice: extra row only in the first 8 steps. CosyVoice 2: the reported runs feed the second stream every step; `run_cosyvoice --lazy` skips it (same clips up to float error on 48 items) | fixed ("is skipped" -> "can be skipped") |
| `rho_t = sigma tau / (1 + (sigma - 1) tau)`, `tau = t/32`, `sigma = 0.1`: 3, 9, 23, 60 % at steps 8, 16, 24, 30 | upstream `_get_time_steps(t_shift=0.1)`; per-step counts use `ceil`, so "about" | ok |
| Excite `alpha_t` (window's share of text attention on the tag), Fig. neglect | `probe_attn_omnivoice` grids (mean over heads and layers) normalized over the target text; `figures.fig_neglect` | ok |
| Excite Eq. gate `s~ = s + b g_t [i in m, j in T_tag]` | `excite.bias_row` / additive mask on the conditional row, steps 1-8 | ok |
| WER capped at 1 per clip | `bench/quality.py` `report` did not cap: the CosyVoice 2 no-guidance mean was 0.159 | fixed in code (`WER_CAP`); every WER/SIM-o number and per-item count in both papers matches the capped report |
| Excite 6.2 numbers (AUC 0.59 / 0.73; 14.5 -> 9.5 %; 1.09x, 13/14, p = .002; layers 8, 12, 13 1.42x; AUC 0.35 -> 0.75) | `figures.neglect_stats`, `fig_neglect`, `fig_layers` | ok |
| Guidance cost: RTF 1.56 -> 1.08 vs 0.93 on 48 items, clips equal up to 4e-4 | `run_cosyvoice --lazy`, one session | ok |
| Fig. example (FlexSED tracks) | `bench/guidance/fig_example.py`, data in `results/flexsed/` (tracks made with VLSP's FlexSED wrapper, not in this repo) | ok |
