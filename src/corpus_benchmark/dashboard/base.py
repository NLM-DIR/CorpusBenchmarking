import math
import re

PALETTE = [
    "#7F77DD",
    "#378ADD",
    "#1D9E75",
    "#D85A30",
    "#639922",
    "#D4537E",
    "#BA7517",
    "#E24B4A",
    "#888780",
]

OV_COLS = {
    "token": "#888780",
    "men_tok": "#1D9E75",
    "mention": "#D85A30",
    "ident": "#7F77DD",
}

BAR_SCALE = 0.65

JOURNAL_TOPIC_ORDER = [
    "Multidisciplinary",
    "Cell & developmental biology",
    "Molecular biology / biochemistry",
    "Genetics/genomics",
    "Neuroscience & neurology",
    "Microbiology/pathogenesis",
    "Pharmacology",
    "Toxicology",
    "Oncology",
    "Public health / health services",
    "Chemistry / Materials Science",
    "Immunology",
    "Psychiatry & psychology",
    "Health disciplines",
    "General biology / anatomy / physiology",
    "General natural sciences",
    "General / internal medicine",
    "Nutrition, metabolism, and food science",
    "Surgery / anesthesia / perioperative",
    "Diagnostics / pathology / radiology",
    "Pediatrics / reproductive / developmental medicine",
    "Clinical specialties by organ system",
]

JOURNAL_TOPIC_COLORS = {
    "Multidisciplinary": "#888780",
    "Cell & developmental biology": "#7F77DD",
    "Molecular biology / biochemistry": "#378ADD",
    "Genetics/genomics": "#6B6ECF",
    "Neuroscience & neurology": "#D4537E",
    "Microbiology/pathogenesis": "#1D9E75",
    "Pharmacology": "#BA7517",
    "Toxicology": "#E24B4A",
    "Oncology": "#D85A30",
    "Public health / health services": "#8CA252",
    "Chemistry / Materials Science": "#5DCAA5",
    "Immunology": "#639922",
    "Psychiatry & psychology": "#AFA9EC",
    "Health disciplines": "#BD9E39",
    "General biology / anatomy / physiology": "#2AA876",
    "General natural sciences": "#4C78A8",
    "General / internal medicine": "#AD494A",
    "Nutrition, metabolism, and food science": "#F2A541",
    "Surgery / anesthesia / perioperative": "#B279A2",
    "Diagnostics / pathology / radiology": "#72B7B2",
    "Pediatrics / reproductive / developmental medicine": "#FF9DA6",
    "Clinical specialties by organ system": "#9D755D",
}

def get_metric(data, metric, field="value", default=None, scope: str | None = None):
    for item in data:
        if item.get("metric_name") == metric:
            source = item
            if scope and scope != "all":
                source = (item.get("scopes") or {}).get(scope)
                if not source:
                    return default
            v = source.get(field, source.get("value", default))
            if v is None:
                return default
            if isinstance(v, float) and math.isnan(v):
                return default
            return v
    return default

def get_stat(data, metric, stat, default=None, scope: str | None = None):
    val = get_metric(data, metric, scope=scope)
    if not isinstance(val, dict):
        return default
    v = val.get(stat, default)
    if v is None:
        return default
    try:
        if math.isnan(float(v)):
            return default
    except (TypeError, ValueError):
        pass
    return v

def norm_corpus_name(s):
    s = s.lower()
    for suf in ("_corpus", "_train", "_test", "_dev"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return re.sub(r"[^a-z0-9]", "", s)
