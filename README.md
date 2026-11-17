# Tag-local guidance for non-verbal vocalization tags in TTS

Training-free, inference-time guidance that makes a TTS model voice its inline non-verbal tags
(`I can't believe it [laughter] you came.`) where they are written. It adds three terms, all of them local to the
tag, to a base model's decoding:

| Term | Setting | What it does |
|---|---|---|
| Presence contrast | `l1` | `+ l1 · m · (c − c_notag)`: log-probabilities with the tag minus those with the tag removed |
| Type contrast | `l2` | `+ l2 · m · (c − c_confuser)`: minus those with the tag swapped for a commonly confused tag |
| Attention excitation | `b` | `+ b` on the attention scores from the speech in the tag's window to the tag's text tokens |

`m` is a window over the output: from the word before the tag to the word after it. How it is found depends on
the model family:

- **OmniVoice** (masked diffusion, fixed length): from OmniVoice's own duration estimate of the text up to each
  point, dilated by 0.3 s. The terms act in the first 8 of 32 decoding steps, alongside OmniVoice's CFG (g = 2).
- **CosyVoice 2** (autoregressive LM + flow matching): the output length is open, so the window is read from the
  model. The current speech token's attention over the text in LM layers 11 and 16 peaks at a text token, and the
  terms act while that token is within the tag's word span (±1 token). The LM has no text CFG, so g = 0.

## Layout

The two methods are separate modules and have separate evidence; the model adapters apply both.

```
src/tagguide/
  contrast.py          guidance: presence and type contrasts on log-probabilities
  excite.py            attention excitation: the additive bias on the scores to the tag's tokens
  omnivoice.py         Guide: OmniVoice's decoding loop with both methods; patch_tokenization()
  cosyvoice.py         Reader (read position, excitation hook), guided_inference for the CosyVoice 2 LM
  tags.py              tag finding, removal and confuser swap
configs/superbench.yaml  settings, per-model window/steps, tag maps and confusers, judge
bench/                 NVV-SuperBench, shared by both methods
  superbench.py        benchmark items, reference voices, config and .env loading
  run_omnivoice.py     synthesis in every setting
  run_cosyvoice.py
  score.py             the benchmark's verifier and metrics (with qwen_judge.py)
  qwen_judge.py        Qwen-Omni judge standing in for Gemini behind the benchmark's unchanged prompt
  outcomes.py          hit / off / other / none split of each judged item
  stats.py             fixed / broken counts, sign test, RTF
  guidance/            README.md: motivation and evidence for the contrasts; evidence.py recomputes it
  excite/              README.md: motivation and evidence for excitation; evidence.py recomputes it
    attn_omnivoice.py, attn_cosyvoice.py, attn_report.py   attention traces (natural vs excited share on the tag)
    control_omnivoice.py   placebo (bias on another word), reseed, excitation in later steps
    read_position_check.py which CosyVoice 2 layers track the text
refs/                  the two fixed reference voices every item is cloned from
results/               judged predictions, metrics, timing and attention traces of the reported runs (no audio)
```

## Setup

The benchmark is a git submodule; the models run in their own environments.

```bash
git clone --recursive <this repo>        # or: git submodule update --init
cp .env.example .env                     # paths, judge key
```

- **OmniVoice**: an environment with [OmniVoice](https://github.com/k2-fsa/OmniVoice) (we used commit `08be0b4`),
  then `pip install -e .` here. `run_omnivoice.load` applies `patch_tokenization()`: upstream `generate()`
  tokenizes tags apart from the surrounding text, unlike training; every setting, base included, runs patched.
- **CosyVoice 2**: a [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) environment (commit `074ca6d`) with the
  CosyVoice2-0.5B model; set `COSYVOICE_REPO` and `COSYVOICE_MODEL_DIR`, then `pip install -e .` here.
- **Scoring**: `python -m venv .venv-score`, then
  `.venv-score/bin/pip install -e . google-generativeai tqdm requests` (the benchmark's `predict_nvc.py` imports
  the Gemini SDK even when `qwen_judge.py` replaces it).

## Running the benchmark

Run from the repository root (`python -m`, so `bench` is importable).

```bash
python -m bench.run_omnivoice --out out           # 150 items x 5 settings; resumes; --per_type N for a subset
python -m bench.run_cosyvoice --out out           # 400 items x 5 settings
python -m bench.score --out out                   # judge + metrics -> out/nvv_pred/<judge>/nvv_report.md
python -m bench.outcomes --out out                # -> out/nvv_pred/<judge>/outcomes.md

# evidence for each method, from judged predictions (and traces); works on results/ without a GPU or judge calls
python -m bench.guidance.evidence --out results   # see bench/guidance/README.md
python -m bench.excite.evidence --out results     # see bench/excite/README.md
```

Settings (`configs/superbench.yaml`): `base` (none), `p2` (l1 = 2), `t3` (l2 = 3), `e2` (b = 2), `p2t3e2` (all).
Each item is synthesized once with seed 0 in every setting, cloning a fixed voice (`refs/`, female for odd ids,
male for even).

### Evaluation

[NVV-SuperBench](https://arxiv.org/abs/2604.16211), tag-based English part (2,250 items, 45 NVV types). A system is
scored on the types its tag inventory covers: OmniVoice 5 types (150 items), CosyVoice 2 eight (400). Its verifier
is told the target type and answers whether it is present and at which word; P / R / F1 use a 2-word collar,
Coverage is the share of all 2,250 items a system covers (fixed by its inventory, unchanged by guidance), CA-F1 the
harmonic mean of F1 and Coverage. The benchmark's judge, `gemini-2.5-pro`, is closed to new API users, so the judge
is `qwen3.8-omni-flash` behind the benchmark's unchanged prompt: absolute numbers are not comparable with the
benchmark paper's tables, only settings under one judge.

## Results

One judge run, seed 0 (`results/`). Fixed / broken: items missed by base and hit by the setting, and the reverse;
p from an exact two-sided sign test. RTF: generation time / audio length on one RTX 5070 Ti (16 GB).

| Model | Setting | Coverage | F1 | CA-F1 | Fixed / broken | p | RTF |
|---|---|---|---|---|---|---|---|
| OmniVoice | base | 0.067 | 0.909 | 0.124 | | | 0.15 |
| | p2 | | 0.967 | | 12 / 2 | 0.013 | |
| | t3 | | 0.966 | | 12 / 4 | 0.077 | |
| | e2 | | 0.970 | 0.125 | 12 / 1 | 0.003 | |
| | p2t3e2 | | 0.970 | 0.125 | 13 / 2 | 0.007 | |
| CosyVoice 2 | base | 0.178 | 0.551 | 0.269 | | | 0.80 |
| | p2 | | 0.601 | | 68 / 45 | 0.038 | 1.33 |
| | t3 | | 0.567 | | 57 / 50 | 0.56 | 1.33 |
| | e2 | | 0.603 | | 61 / 38 | 0.027 | 0.82 |
| | p2t3e2 | | 0.657 | 0.280 | 91 / 44 | 6e-5 | 1.86 |

Excitation controls on OmniVoice (F1; fixed / broken against base): excitation on the word before the tag 0.916
(8 / 8), on the sentence's first word 0.926 (6 / 4), base with seed 1 0.943 (11 / 5), against e2's 0.970 (12 / 1).
The gain needs the tag's tokens, but a reseed alone fixes most base misses, so one seed overstates the effect size.

## Prior work this builds on

The excitation term is an additive pre-softmax bias confined to a window, as in eDiff-I's paint-with-words
([Balaji et al. 2022](https://arxiv.org/abs/2211.01324)) and DenseDiffusion
([Kim et al. 2023](https://arxiv.org/abs/2308.12964)) for image regions; Attend-and-Excite
([Chefer et al. 2023](https://arxiv.org/abs/2301.13826)) instead optimizes the latent so a neglected token's peak
attention rises. Here the window is found by the TTS model itself rather than given by the user.
