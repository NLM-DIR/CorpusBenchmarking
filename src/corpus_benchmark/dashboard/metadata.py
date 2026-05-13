import json
from .base import norm_corpus_name

def _topic_distribution(td_raw):
    topic_clean = {
        k: float(v)
        for k, v in (td_raw or {}).items()
        if k not in ("Unknown", None) and v
    }
    return (
        {topic: round(frac * 100, 1) for topic, frac in sorted(topic_clean.items())}
        if topic_clean
        else None
    )


def _process_metadata(jd_raw, yd_raw, journal_td_raw, article_td_raw):
    j_clean = {
        k: v for k, v in (jd_raw or {}).items() if k not in ("Unknown", None) and v
    }

    if not j_clean:
        journal = None
    else:
        sj = sorted(j_clean.items(), key=lambda x: -x[1])
        journal = {
            "n_journals": len(j_clean),
            "top1_name": sj[0][0],
            "top1_pct": round(sj[0][1] * 100, 1),
            "top3_pct": round(sum(v for _, v in sj[:3]) * 100, 1),
        }

    y_clean = {}
    for k, v in (yd_raw or {}).items():
        if k not in ("Unknown", None):
            try:
                y_clean[int(k)] = float(v)
            except (ValueError, TypeError):
                pass

    if not y_clean:
        year = None
    else:
        decades = {}
        for yr, frac in y_clean.items():
            d = (yr // 10) * 10
            decades[d] = round(decades.get(d, 0) + frac * 100, 1)
        year = {
            "year_min": min(y_clean),
            "year_max": max(y_clean),
            "span": max(y_clean) - min(y_clean),
            "mode_year": max(y_clean, key=lambda yr: y_clean[yr]),
            "decades": decades,
            "year_pcts": {
                yr: round(frac * 100, 2) for yr, frac in sorted(y_clean.items())
            },
        }

    return {
        "journal": journal,
        "year": year,
        "topic_dist": _topic_distribution(journal_td_raw),
        "article_topic_dist": _topic_distribution(article_td_raw),
        "has_metadata": journal is not None or year is not None,
    }


def load_metadata_stats(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for corpus_name, metrics in raw.items():
        jd = next(
            (
                m.get("value", {})
                for m in metrics
                if m.get("metric_name") == "journal_distribution"
            ),
            {},
        )
        yd = next(
            (
                m.get("value", {})
                for m in metrics
                if m.get("metric_name") == "publication_year_distribution"
            ),
            {},
        )
        journal_td = next(
            (
                m.get("value", {})
                for m in metrics
                if m.get("metric_name") == "journal_MeSH_topic_distribution"
            ),
            {},
        )
        article_td = next(
            (
                m.get("value", {})
                for m in metrics
                if m.get("metric_name") == "article_MeSH_topic_distribution"
            ),
            {},
        )
        result[norm_corpus_name(corpus_name)] = _process_metadata(jd, yd, journal_td, article_td)
    return result


def attach_metadata_to_corpora(corpora, metadata):
    for c in corpora:
        c["metadata"] = metadata.get(norm_corpus_name(c["raw_name"]))
