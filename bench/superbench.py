"""Shared pieces for running NVV-SuperBench (tag-based, English) with tag-local guidance.

NVV-SuperBench (arXiv 2604.16211, https://github.com/lmxue/NVV-SuperBench) writes each target NVV as an inline
`<type>` marker in `text_with_mark`. A system is only scored on the types its own tag inventory covers (the
benchmark's coverage), so every system maps benchmark types to its native tags (configs/superbench.yaml).

Paths and keys come from the environment or from <repo>/.env (loaded here, without overriding the environment):
NVV_SUPERBENCH (benchmark checkout, default third_party/NVV-SuperBench), OMNIVOICE_CKPT, COSYVOICE_REPO,
COSYVOICE_MODEL_DIR, QWEN_API_KEY, QWEN_BASE_URL.
"""
import json
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REFS = ROOT / "refs"


def load_dotenv(path=ROOT / ".env"):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
BENCH = Path(os.environ.get("NVV_SUPERBENCH", ROOT / "third_party" / "NVV-SuperBench"))
CONFIG = yaml.safe_load((ROOT / "configs" / "superbench.yaml").read_text(encoding="utf-8"))
SETTINGS = {name: tuple(v) for name, v in CONFIG["settings"].items()}
MARK_RE = re.compile(r"<[^<>]+>")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_rows(out):
    """(system, id) pairs already synthesized, from <out>/timing.jsonl, so a stopped run resumes."""
    log = Path(out) / "timing.jsonl"
    return {(r["system"], r["id"]) for r in read_jsonl(log)} if log.exists() else set()


def tidy(text):
    """Collapse the spaces a removed or replaced marker leaves, and spaces before punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.!?;:])", r"\1", text)


def items(system, lang="en", types=None, per_type=0):
    """Benchmark items whose type the system covers, with the system's native tag in `text`.

    Returns dicts: id, type, tag, text (native tag), notag (tag removed), confuser (tag replaced by its confuser).
    """
    tagmap, confuser = CONFIG[system]["tags"], CONFIG[system]["confuser"]
    rows = json.loads((BENCH / "dataset" / f"nvbench_data_{lang}.json").read_text(encoding="utf-8"))
    out, seen = [], {}
    for r in rows:
        kind = r["non_verbal_events"][0]
        if kind not in tagmap or (types and kind not in types):
            continue
        if per_type and seen.get(kind, 0) >= per_type:
            continue
        seen[kind] = seen.get(kind, 0) + 1
        tag = tagmap[kind]
        out.append({
            "id": r["id"], "type": kind, "tag": tag,
            "text": tidy(MARK_RE.sub(f" {tag} ", r["text_with_mark"])),
            "notag": tidy(MARK_RE.sub(" ", r["text_with_mark"])),
            "confuser": tidy(MARK_RE.sub(f" {confuser[tag]} ", r["text_with_mark"])),
        })
    return out


def ref_for(item_id):
    """Fixed reference voice per item: female for odd ids, male for even (refs/make_refs.py)."""
    n = int(item_id.split("_")[-1])
    name = "female" if n % 2 else "male"
    return REFS / f"{name}.wav", (REFS / f"{name}.txt").read_text(encoding="utf-8").strip()


def env_path(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"{name} not set (environment or .env)")
    return value
