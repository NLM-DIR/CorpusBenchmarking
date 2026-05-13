import json
import math
from .base import get_metric, get_stat

def compute_entropy(data, scope: str | None = None):
    dist = get_metric(data, "label_distribution", scope=scope) or {}
    probs = [v for v in dist.values() if v and v > 0]
    return -sum(p * math.log2(p) for p in probs)

def compute_entropy_from_counts(counts):
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = [v / total for v in counts.values() if v and v > 0]
    return -sum(p * math.log2(p) for p in probs)

def get_id_info(data, scope: str | None = None):
    dist = get_metric(data, "identifier_resource_distribution", scope=scope) or {}
    named = sorted([k for k in dist if k not in ("null", "<NIL>", None)])
    null_frac = dist.get("null", 0) + dist.get("<NIL>", 0)
    if not named:
        return dict(has_ids=False, partial=False, label="none", css_class="no")
    if null_frac > 0.05:
        return dict(
            has_ids=True,
            partial=True,
            label=f"{', '.join(named)} (partial)",
            css_class="part",
        )
    return dict(has_ids=True, partial=False, label=", ".join(named), css_class="yes")

def get_total_ann(data, scope: str | None = None):
    details = get_metric(data, "label_distribution", "details", scope=scope) or {}
    counts = details.get("counts", {})
    if counts:
        return sum(counts.values())
    apd = get_stat(data, "annotations_per_document_stats", "mean", 0)
    dc = get_metric(data, "document_count", default=0)
    return int(round(apd * dc))

def summarise_corpus(name, data):
    ld = get_metric(data, "label_distribution") or {}
    label_counts = (get_metric(data, "label_distribution", "details") or {}).get("counts", {})
    info = get_id_info(data)
    return dict(
        name=name.replace("_corpus", "").replace("_", "-"),
        raw_name=name,
        metric_results=data,
        doc_count=get_metric(data, "document_count", default=0),
        token_count=get_metric(data, "token_count", default=0),
        n_types=len(ld),
        types=list(ld.keys()),
        label_counts=label_counts,
        entropy=round(compute_entropy(data), 2),
        total_ann=get_total_ann(data),
        ann_per_doc=round(get_stat(data, "annotations_per_document_stats", "mean", 0), 2),
        ann_per_1k=round(
            get_stat(data, "annotations_per_1000_tokens_stats", "mean", 0), 2
        ),
        men_per_doc=round(
            get_stat(data, "unique_mentions_per_document_stats", "mean", 0), 2
        ),
        ids_per_doc=round(
            get_stat(data, "unique_identifiers_per_document_stats", "mean", 0), 2
        ),
        ambiguity=round(get_stat(data, "ambiguity_degree_stats", "mean", 1.0), 3),
        variation=get_stat(data, "variation_degree_stats", "mean"),
        id_vocab=info["label"],
        id_class=info["css_class"],
        has_ids=info["has_ids"],
        overlap=None,
        metadata=None,
    )

def load_corpora_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [summarise_corpus(name, data) for name, data in raw.items()]
