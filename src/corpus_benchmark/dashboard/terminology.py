import json
from .base import norm_corpus_name

def _term_label(name: str) -> str:
    return {
        "mesh": "MeSH",
        "cell_ontology": "Cell Ontology",
        "mondo": "MONDO",
        "chebi": "ChEBI",
    }.get(name, name.replace("_", " "))

def _scope_label(scope: str) -> str:
    return scope.replace("_", " ").title()

def _metric_scope_payload(metric: dict, scope: str) -> dict | None:
    if scope == "all":
        return metric
    return (metric.get("scopes") or {}).get(scope)

def _process_term_payload(corpus_name: str, terminology_name: str, hlc: dict, dc: dict) -> dict:
    details = hlc.get("details", {})
    n_in = details.get("n_input_ids", 0)
    n_miss = details.get("n_missing_ids", 0)
    miss_ids = details.get("missing_ids", [])
    branches = {}
    for item in hlc.get("value", []):
        code = item.get("branch_code")
        if not code:
            continue
        branches[code] = {
            "label": item.get("label") or code,
            "count": item.get("count", 0),
            "annotation_count": item.get("annotation_count", 0),
            "proportion": item.get("proportion", 0) or 0,
            "annotation_proportion": item.get("annotation_proportion", 0) or 0,
            "total": item.get("terminology_total_count", item.get("mesh_total_count", 0)),
            "configured_anchor": bool(details.get("term_overrides_path")),
        }

    depth = {}
    mean_num = 0.0
    mean_den = 0.0
    for item in dc.get("value", []):
        d = str(item.get("depth"))
        count = item.get("count", 0) or 0
        depth[d] = {
            "count": count,
            "proportion": item.get("proportion", 0) or 0,
            "total": item.get("terminology_total_count", item.get("mesh_total_count", 0)),
        }
        try:
            mean_num += float(d) * float(count)
            mean_den += float(count)
        except (TypeError, ValueError):
            pass

    return {
        "display_name": corpus_name.replace("_", "-").replace("_corpus", ""),
        "entity_scope": "",
        "entity_scope_label": "",
        "terminology": terminology_name,
        "terminology_label": _term_label(terminology_name),
        "series_label": f"{corpus_name.replace('_corpus', '').replace('_', '-')} / {_term_label(terminology_name)}",
        "n_input_ids": n_in,
        "n_missing_ids": n_miss,
        "unique_missing": len(set(miss_ids)),
        "coverage_pct": round((n_in - n_miss) / n_in * 100, 2) if n_in > 0 else 0,
        "missing_pct": round(n_miss / n_in * 100, 2) if n_in > 0 else 0,
        "branches": branches,
        "depth": depth,
        "mean_depth": round(mean_num / mean_den, 2) if mean_den else 0,
    }

def process_terminology_stats(raw):
    processed = {}
    for corpus_name, metrics in raw.items():
        if not isinstance(metrics, list):
            continue
        high_metrics = [m for m in metrics if m.get("metric_name") == "high_level_concept_counts"]
        depth_metrics = [m for m in metrics if m.get("metric_name") == "concept_depth_counts"]
        scope_keys = {"all"}
        for metric in high_metrics + depth_metrics:
            scope_keys.update((metric.get("scopes") or {}).keys())
        by_scope = {scope: [] for scope in scope_keys}
        for high_metric in high_metrics:
            terminology_name = (high_metric.get("details") or {}).get("terminology")
            if not terminology_name:
                continue
            depth_metric = next(
                (
                    m
                    for m in depth_metrics
                    if (m.get("details") or {}).get("terminology") == terminology_name
                ),
                {},
            )
            for scope in scope_keys:
                high_payload = _metric_scope_payload(high_metric, scope)
                depth_payload = _metric_scope_payload(depth_metric, scope) if depth_metric else None
                if not high_payload or not depth_payload:
                    continue
                entry = _process_term_payload(corpus_name, terminology_name, high_payload, depth_payload)
                if entry["n_input_ids"] > 0:
                    entry["entity_scope"] = scope
                    entry["entity_scope_label"] = _scope_label(scope)
                    by_scope[scope].append(entry)
        processed[norm_corpus_name(corpus_name)] = {"by_scope": by_scope}
    return processed

def attach_terminology_to_corpora(corpora, term_data):
    for c in corpora:
        c["terminology"] = term_data.get(norm_corpus_name(c["raw_name"]))

def load_terminology_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
