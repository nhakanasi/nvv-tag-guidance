"""Two fixed English reference voices (female.wav, male.wav + transcripts) for the voice-cloning systems.

NVV-SuperBench gives no reference audio. OmniVoice and CosyVoice 2 clone a voice, so every item is prompted with
one of these two clips (superbench.ref_for: female for odd item ids, male for even). They come from OmniVoice's
voice-design mode, contain no NVV, and were made once with seed 0. The committed clips are the ones used for the
results; rerunning this script may not reproduce them bit for bit on other hardware.

    python refs/make_refs.py
"""
import os
from pathlib import Path

import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice

HERE = Path(__file__).resolve().parent
TEXT = "The weather report said it would rain this afternoon, so I brought an umbrella with me to work."
VOICES = {"female": "female, american accent", "male": "male, american accent"}

model = OmniVoice.from_pretrained(os.environ.get("OMNIVOICE_CKPT", "k2-fsa/OmniVoice"), device_map="cuda",
                                  dtype=torch.float16)
for name, instruct in VOICES.items():
    torch.manual_seed(0)
    audio = model.generate(text=TEXT, language="en", instruct=instruct)[0]
    sf.write(HERE / f"{name}.wav", audio.squeeze(), model.sampling_rate)
    (HERE / f"{name}.txt").write_text(TEXT + "\n", encoding="utf-8")
    print(name, audio.shape[-1] / model.sampling_rate, "s")
