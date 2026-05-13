import json
import math
import re
from .base import norm_corpus_name

def _ov_val(metrics, name, scope: str | None = None):
    for m in metrics:
        if m["metric_name"] == name:
            source = m
            if scope and scope != "all":
                source = (m.get("scopes") or {}).get(scope)
                if not source:
                    return None
            v = source.get("value")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            return v
    return None

def _split_sizes(metrics):
    for m in metrics:
        if m["metric_name"] == "token_overlap":
            d = m.get("details", {})
            tr = next((v for k, v in d.items() if "train" in k.lower()), 0)
            te = next(
                (v for k, v in d.items() if "test" in k.lower() or "dev" in k.lower()),
                0,
            )
            return int(tr), int(te)
    return 0, 0

def _corpus_from_key(key):
    m = re.match(r"\((\w+?)_(?:train|test|dev)", key)
    return m.group(1) if m else key.strip("()")

def load_overlaps_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for key, metrics in raw.items():
        nk = norm_corpus_name(_corpus_from_key(key))
        tr, te = _split_sizes(metrics)
        scope_keys = sorted(
            {
                scope_key
                for metric in metrics
                for scope_key in (metric.get("scopes") or {})
            }
        )
        scopes = {}
        for scope_key in scope_keys:
            scopes[scope_key] = {
                "token_overlap": _ov_val(metrics, "token_overlap"),
                "mention_token_overlap": _ov_val(metrics, "mention_token_overlap", scope_key),
                "mention_overlap": _ov_val(metrics, "mention_overlap", scope_key),
                "identifier_overlap": _ov_val(metrics, "identifier_overlap", scope_key),
                "train_size": tr,
                "test_size": te,
            }
        result[nk] = {
            "token_overlap": _ov_val(metrics, "token_overlap"),
            "mention_token_overlap": _ov_val(metrics, "mention_token_overlap"),
            "mention_overlap": _ov_val(metrics, "mention_overlap"),
            "identifier_overlap": _ov_val(metrics, "identifier_overlap"),
            "train_size": tr,
            "test_size": te,
            "scopes": scopes,
        }
    return result

def attach_overlaps_to_corpora(corpora, overlaps):
    for c in corpora:
        c["overlap"] = overlaps.get(norm_corpus_name(c["raw_name"]))
