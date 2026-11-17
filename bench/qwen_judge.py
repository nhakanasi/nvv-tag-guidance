"""Run NVV-SuperBench's predict_nvc.py with a Qwen-Omni judge (Alibaba Model Studio) instead of Gemini.

predict_nvc.py talks to Gemini through `google.generativeai`. This script puts a minimal stand-in for that module
in sys.modules, then runs predict_nvc.py unchanged, so its prompt, format checks and re-asks, position logic and
output JSON stay the benchmark's; only the model answering the prompt changes. Use it with --input_mode bytes (the
stand-in has no file upload).

Model Studio's OpenAI-compatible endpoint (QWEN_BASE_URL; the international DashScope endpoint by default) takes the
audio inline as base64 `input_audio`, and Omni models only answer streamed requests; the streamed text is joined
into one reply. The key is QWEN_API_KEY.

    python bench/qwen_judge.py <audio_dir> --gt_json ... --model_name qwen3.8-omni-flash --input_mode bytes [...]
"""
import base64
import json
import os
import random
import runpy
import sys
import time
import types

import requests

import superbench as sb

PREDICT = sb.BENCH / "evaluation" / "objective_evaluation" / "nvv_precision_recall_f1_distance" / "predict_nvc.py"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
RETRY_STATUS = (429, 500, 502, 503, 504)


class Response:
    def __init__(self, text):
        self.text = text


class GenerativeModel:
    def __init__(self, model_name):
        self.model = model_name.split("/")[-1]

    def generate_content(self, contents, generation_config=None, **_):
        content = []
        for part in contents:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            else:  # {"mime_type": ..., "data": bytes} from predict_nvc.py's bytes mode
                fmt = part["mime_type"].split("/")[-1].replace("x-", "").replace("mpeg", "mp3")
                data = "data:;base64," + base64.b64encode(part["data"]).decode()
                content.append({"type": "input_audio", "input_audio": {"data": data, "format": fmt}})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "modalities": ["text"], "stream": True, "stream_options": {"include_usage": True},
                "temperature": (generation_config or {}).get("temperature", 0.0)}
        key = os.environ["QWEN_API_KEY"]
        url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
        resp = None
        for attempt in range(8):
            try:
                resp = requests.post(url, json=body, headers={"Authorization": f"Bearer {key}"}, stream=True,
                                     timeout=300)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.ok:
                text = []
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    for choice in json.loads(line[5:]).get("choices") or []:
                        text.append((choice.get("delta") or {}).get("content") or "")
                return Response("".join(text))
            if resp is not None and resp.status_code not in RETRY_STATUS:
                raise RuntimeError(f"Model Studio {resp.status_code}: {resp.text[:500]}")
            time.sleep(min(60.0, 2.0 * 2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f"Model Studio: gave up after retries ({resp.status_code if resp is not None else 'network'})")


def unsupported(*_, **__):
    raise RuntimeError("file upload is not supported with the Qwen judge: use --input_mode bytes")


stub = types.ModuleType("google.generativeai")
stub.configure = lambda **_: None
stub.GenerativeModel = GenerativeModel
stub.upload_file = unsupported
stub.delete_file = lambda *_: None
google = sys.modules.setdefault("google", types.ModuleType("google"))
google.generativeai = stub
sys.modules["google.generativeai"] = stub

if __name__ == "__main__":
    if not os.environ.get("QWEN_API_KEY"):
        sys.exit("QWEN_API_KEY not set (environment or .env)")
    os.environ.setdefault("GEMINI_API_KEY", "unused")  # predict_nvc.py refuses to start without one
    sys.argv = [str(PREDICT)] + sys.argv[1:]
    runpy.run_path(str(PREDICT), run_name="__main__")
