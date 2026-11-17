"""Per-item comparisons of judged settings: hits, fixed / broken counts and the exact sign test, F1 and RTF."""
import json
from math import comb
from pathlib import Path

from bench import superbench as sb
from bench.outcomes import outcome


class Judged:
    """Judged predictions under <out>/nvv_pred/<judge>/ (run 1), plus timing from <out>/timing.jsonl."""

    def __init__(self, out, judge=None, run=1, delta=None):
        self.out = Path(out)
        self.root = self.out / "nvv_pred" / (judge or sb.CONFIG["judge"]["model"])
        self.run, self.delta = run, delta if delta is not None else sb.CONFIG["judge"]["delta"]
        self._preds = {}

    def has(self, system):
        return (self.root / f"run{self.run}" / system / f"{system}.json").exists()

    def preds(self, system):
        if system not in self._preds:
            path = self.root / f"run{self.run}" / system / f"{system}.json"
            self._preds[system] = {p["id"]: p for p in json.loads(path.read_text(encoding="utf-8")).values()}
        return self._preds[system]

    def outcomes(self, system):
        return {i: outcome(p, self.delta) for i, p in self.preds(system).items()}

    def hits(self, system):
        return {i: o == "hit" for i, o in self.outcomes(system).items()}

    def metric(self, system, key="f1_paired"):
        m = json.loads((self.root / f"{system}.metrics.json").read_text(encoding="utf-8"))
        vals = [r[key] for r in m["runs"] if r.get(key) is not None]
        return sum(vals) / len(vals)

    def extra_nvv(self, system):
        """Clips in which the judge heard an NVV that was not asked for."""
        return sum(bool(p.get("hallucinated_events")) for p in self.preds(system).values())

    def rtf(self, system):
        """Total generation time / total audio length over the system's clips."""
        rows = [r for r in sb.read_jsonl(self.out / "timing.jsonl") if r["system"] == system]
        return sum(r["gen_s"] for r in rows) / sum(r["out_s"] for r in rows) if rows else float("nan")


def sign_test(a, b):
    """Exact two-sided sign test on a successes against b."""
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def paired(ha, hb):
    """(items hit in b but not a, items hit in a but not b, p): fixed / broken when a is the reference."""
    better = sum(hb[i] and not ha[i] for i in ha)
    worse = sum(ha[i] and not hb[i] for i in ha)
    return better, worse, sign_test(better, worse)


def fmt_p(p):
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0") if p < 1 else "1"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])
