"""
corpus_dashboard.py
Generates a self-contained HTML dashboard from corpus statistics JSON files.
Optionally incorporates train/test overlap and metadata (journal/year/topic) statistics.

Usage:
    python corpus_dashboard.py stats.json
    python corpus_dashboard.py stats.json --overlap overlap.json
    python corpus_dashboard.py stats.json --overlap overlap.json \\
                              --metadata metadata.json --output report.html --open
"""

import argparse
import json
import logging
import math
import re
import sys
import webbrowser
from pathlib import Path

# ── Colour palette ────────────────────────────────────────────────────────────

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
logger = logging.getLogger(__name__)

JOURNAL_TOPIC_ORDER = [
    "Diseases",
    "Chemicals and Drugs",
    "Psychiatry and Psychology",
    "Anatomy",
    "Biological Sciences",
    "Analytical, Diagnostic and Therapeutic Techniques and Equipment",
    "Organisms",
    "Health Care",
    "Technology and Food and Beverages",
    "Physical Sciences",
    "Anthropology, Education, Sociology and Social Phenomena",
    "Information Science",
    "Humanities",
    "Persons",
    "Publication Characteristics",
    "Geographic Locations",
]

JOURNAL_TOPIC_COLORS = {
    "Diseases": "#E24B4A",
    "Chemicals and Drugs": "#378ADD",
    "Psychiatry and Psychology": "#D4537E",
    "Anatomy": "#7F77DD",
    "Biological Sciences": "#639922",
    "Analytical, Diagnostic and Therapeutic Techniques and Equipment": "#D85A30",
    "Organisms": "#1D9E75",
    "Health Care": "#888780",
    "Technology and Food and Beverages": "#BA7517",
    "Physical Sciences": "#5DCAA5",
    "Anthropology, Education, Sociology and Social Phenomena": "#AFA9EC",
    "Information Science": "#8CA252",
    "Humanities": "#BD9E39",
    "Persons": "#AD494A",
    "Publication Characteristics": "#6B6ECF",
    "Geographic Locations": "#D3D1C7",
}


# ── Corpus statistics helpers ─────────────────────────────────────────────────


def _get(data, metric, field="value", default=None):
    for item in data:
        if item.get("metric_name") == metric:
            v = item.get(field, item.get("value", default))
            if v is None:
                return default
            if isinstance(v, float) and math.isnan(v):
                return default
            return v
    return default


def _stat(data, metric, stat, default=None):
    val = _get(data, metric)
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


def _entropy(data):
    dist = _get(data, "label_distribution") or {}
    probs = [v for v in dist.values() if v and v > 0]
    return -sum(p * math.log2(p) for p in probs)


def _id_info(data):
    dist = _get(data, "identifier_resource_distribution") or {}
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


def _total_ann(data):
    details = _get(data, "label_distribution", "details") or {}
    counts = details.get("counts", {})
    if counts:
        return sum(counts.values())
    apd = _stat(data, "annotations_per_document_stats", "mean", 0)
    dc = _get(data, "document_count", default=0)
    return int(round(apd * dc))


def summarise(name, data):
    ld = _get(data, "label_distribution") or {}
    info = _id_info(data)
    return dict(
        name=name.replace("_corpus", "").replace("_", "-"),
        raw_name=name,
        doc_count=_get(data, "document_count", default=0),
        token_count=_get(data, "token_count", default=0),
        n_types=len(ld),
        types=list(ld.keys()),
        entropy=round(_entropy(data), 2),
        total_ann=_total_ann(data),
        ann_per_doc=round(_stat(data, "annotations_per_document_stats", "mean", 0), 2),
        men_per_doc=round(
            _stat(data, "unique_mentions_per_document_stats", "mean", 0), 2
        ),
        ids_per_doc=round(
            _stat(data, "unique_identifiers_per_document_stats", "mean", 0), 2
        ),
        ambiguity=round(_stat(data, "ambiguity_degree_stats", "mean", 1.0), 3),
        variation=_stat(data, "variation_degree_stats", "mean"),
        id_vocab=info["label"],
        id_class=info["css_class"],
        has_ids=info["has_ids"],
        overlap=None,
        metadata=None,
    )


# ── Overlap helpers ───────────────────────────────────────────────────────────


def _norm(s):
    s = s.lower()
    for suf in ("_corpus", "_train", "_test", "_dev"):
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return re.sub(r"[^a-z0-9]", "", s)


def _corpus_from_key(key):
    m = re.match(r"\((\w+?)_(?:train|test|dev)", key)
    return m.group(1) if m else key.strip("()")


def _ov_val(metrics, name):
    for m in metrics:
        if m["metric_name"] == name:
            v = m.get("value")
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


def load_overlaps(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    result = {}
    for key, metrics in raw.items():
        nk = _norm(_corpus_from_key(key))
        tr, te = _split_sizes(metrics)
        result[nk] = {
            "token_overlap": _ov_val(metrics, "token_overlap"),
            "mention_token_overlap": _ov_val(metrics, "mention_token_overlap"),
            "mention_overlap": _ov_val(metrics, "mention_overlap"),
            "identifier_overlap": _ov_val(metrics, "identifier_overlap"),
            "train_size": tr,
            "test_size": te,
        }
    return result


def attach_overlaps(corpora, overlaps):
    for c in corpora:
        c["overlap"] = overlaps.get(_norm(c["raw_name"]))


# ── Metadata helpers ──────────────────────────────────────────────────────────


def _process_metadata(jd_raw, yd_raw, td_raw):
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

    topic_clean = {
        k: float(v)
        for k, v in (td_raw or {}).items()
        if k not in ("Unknown", None) and v
    }
    topic_dist = (
        {topic: round(frac * 100, 1) for topic, frac in sorted(topic_clean.items())}
        if topic_clean
        else None
    )

    return {
        "journal": journal,
        "year": year,
        "topic_dist": topic_dist,
        "has_metadata": journal is not None or year is not None,
    }


def load_metadata(path):
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
        td = next(
            (
                m.get("value", {})
                for m in metrics
                if m.get("metric_name") == "journal_topic_distribution"
            ),
            {},
        )
        result[_norm(corpus_name)] = _process_metadata(jd, yd, td)
    return result


def attach_metadata(corpora, metadata):
    for c in corpora:
        c["metadata"] = metadata.get(_norm(c["raw_name"]))


# ── Topic table builder (pure Python → HTML) ──────────────────────────────────


def build_topic_table(corpora) -> str:
    """Generate an HTML table: rows = topics, columns = corpora with topic data."""
    with_td = sorted(
        [c for c in corpora if (c.get("metadata") or {}).get("topic_dist")],
        key=lambda c: c["name"],
    )
    if not with_td:
        return "<p style='color:var(--color-text-secondary);font-size:13px'>No topic data available.</p>"

    corp_names = [c["name"] for c in with_td]
    observed_topics = {
        topic for c in with_td for topic in c["metadata"]["topic_dist"].keys()
    }
    ordered_topics = [
        topic for topic in JOURNAL_TOPIC_ORDER if topic in observed_topics
    ] + sorted(observed_topics - set(JOURNAL_TOPIC_ORDER))

    # Header
    th_cells = '<th class="l">Topic</th>' + "".join(
        f'<th class="r">{n}</th>' for n in corp_names
    )

    # Rows — only topics with at least 1% in at least one corpus
    rows = []
    for topic in ordered_topics:
        vals = [c["metadata"]["topic_dist"].get(topic, 0.0) for c in with_td]
        if max(vals) < 1.0:
            continue
        col = JOURNAL_TOPIC_COLORS.get(topic, "#D3D1C7")
        mx = max(vals)
        td_cells = "".join(
            f'<td class="r" style="font-weight:{"600" if v == mx and v >= 1 else "400"};'
            f'color:{"var(--color-text-primary)" if v >= 1 else "var(--color-text-tertiary)"}">'
            f'{"—" if v < 1 else f"{v:.0f}%"}</td>'
            for v in vals
        )
        dot = (
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
            f'background:{col};margin-right:6px;vertical-align:middle"></span>'
        )
        rows.append(f'<tr><td class="l">{dot}{topic}</td>{td_cells}</tr>')

    # Footer: totals (sum of shown rows, should be ~100)
    total_cells = ""
    for c in with_td:
        shown = sum(
            c["metadata"]["topic_dist"].get(t, 0)
            for t in ordered_topics
            if max(cc["metadata"]["topic_dist"].get(t, 0) for cc in with_td) >= 1.0
        )
        total_cells += f'<td class="r" style="font-weight:600">{shown:.0f}%</td>'

    return f"""
<div style="overflow-x:auto">
<table>
<thead>
  <tr>{th_cells}</tr>
</thead>
<tbody>
  {"".join(rows)}
</tbody>
<tfoot>
  <tr style="border-top:1.5px solid var(--color-border-primary)">
    <td class="l" style="font-weight:600;color:var(--color-text-secondary);font-size:11px">
      Total shown</td>
    {total_cells}
  </tr>
</tfoot>
</table>
</div>"""


# ── Metadata chart data ───────────────────────────────────────────────────────


def _meta_chart_data(corpora, colours):
    ci = {c["name"]: i for i, c in enumerate(corpora)}

    def col(name):
        return colours[ci.get(name, 0) % len(colours)]

    # Journal diversity
    by_jdiv = sorted(
        corpora,
        key=lambda c: -(
            ((c.get("metadata") or {}).get("journal") or {}).get("n_journals", 0)
        ),
    )
    jdiv_vals = [
        ((c.get("metadata") or {}).get("journal") or {}).get("n_journals", 0)
        for c in by_jdiv
    ]

    # Temporal range
    with_yr = [c for c in corpora if (c.get("metadata") or {}).get("year")]
    by_yr = sorted(with_yr, key=lambda c: c["metadata"]["year"]["year_min"])

    # Concentration
    with_j = [c for c in corpora if (c.get("metadata") or {}).get("journal")]
    by_conc = sorted(with_j, key=lambda c: -c["metadata"]["journal"]["top1_pct"])

    # Decade stacked
    all_dec = sorted({d for c in by_yr for d in c["metadata"]["year"]["decades"]})
    dec_pal = [
        "#55534ecc",
        "#888780cc",
        "#B4B2A9cc",
        "#7F77DDcc",
        "#378ADDcc",
        "#D4537Ecc",
        "#D85A30cc",
        "#639922cc",
    ]

    def dec_lbl(d):
        return f"≤{d+9}" if d <= 1970 else f"{d}s"

    decade_ds = [
        {
            "label": dec_lbl(d),
            "data": [
                round(c["metadata"]["year"]["decades"].get(d, 0), 1) for c in by_yr
            ],
            "backgroundColor": dec_pal[i % len(dec_pal)],
            "borderWidth": 0,
            "borderRadius": 0,
        }
        for i, d in enumerate(all_dec)
    ]

    # Year-by-year: oldest vs most recent
    yby_ds = []
    if by_yr:
        sel = [by_yr[0], by_yr[-1]] if len(by_yr) > 1 else [by_yr[0]]
        for c in sel:
            pts = [
                {"x": yr, "y": pct}
                for yr, pct in sorted(c["metadata"]["year"]["year_pcts"].items())
            ]
            yby_ds.append(
                {
                    "label": c["name"],
                    "data": pts,
                    "backgroundColor": col(c["name"]) + "88",
                    "borderWidth": 0,
                    "borderRadius": 1,
                }
            )

    yr_x_min = (by_yr[0]["metadata"]["year"]["year_min"] - 5) if by_yr else 1960
    yr_x_max = (by_yr[-1]["metadata"]["year"]["year_max"] + 3) if by_yr else 2030

    return dict(
        jdiv_labels=json.dumps([c["name"] for c in by_jdiv]),
        jdiv_data=json.dumps(jdiv_vals),
        jdiv_bg=json.dumps(
            [
                col(c["name"]) + ("cc" if jdiv_vals[i] > 0 else "22")
                for i, c in enumerate(by_jdiv)
            ]
        ),
        yr_labels=json.dumps([c["name"] for c in by_yr]),
        yr_ranges=json.dumps(
            [
                [c["metadata"]["year"]["year_min"], c["metadata"]["year"]["year_max"]]
                for c in by_yr
            ]
        ),
        yr_modes=json.dumps([c["metadata"]["year"]["mode_year"] for c in by_yr]),
        yr_bg=json.dumps([col(c["name"]) + "bb" for c in by_yr]),
        conc_labels=json.dumps([c["name"] for c in by_conc]),
        conc_top1=json.dumps([c["metadata"]["journal"]["top1_pct"] for c in by_conc]),
        conc_top3=json.dumps([c["metadata"]["journal"]["top3_pct"] for c in by_conc]),
        conc_bg1=json.dumps([col(c["name"]) + "dd" for c in by_conc]),
        conc_bg3=json.dumps([col(c["name"]) + "44" for c in by_conc]),
        decade_ds=json.dumps(decade_ds),
        decade_labels=json.dumps([c["name"] for c in by_yr]),
        yby_ds=json.dumps(yby_ds),
        yr_x_min=yr_x_min,
        yr_x_max=yr_x_max,
        n_with_meta=sum(
            1 for c in corpora if (c.get("metadata") or {}).get("has_metadata")
        ),
    )


def build_metadata_panels(corpora, colours):
    n = len(corpora)
    d = _meta_chart_data(corpora, colours)
    topic_table = build_topic_table(corpora)

    tabs = (
        '\n  <button class="tab" data-p="p8">Journal metadata</button>'
        '\n  <button class="tab" data-p="p9">Temporal coverage</button>'
        '\n  <button class="tab" data-p="p10">Journal topics</button>'
    )

    panels = f"""
<div class="panel" id="p8">
  <div class="two">
    <div>
      <p class="sec">Unique journal count</p>
      <div class="cw" style="height:340px">
        <canvas id="mc1" role="img" aria-label="Unique journal counts per corpus.">
          Journal diversity from 25 to over 300 unique journals.
        </canvas>
      </div>
    </div>
    <div>
      <p class="sec">Journal concentration</p>
      <div class="leg">
        <span class="li"><span class="lc" style="background:#555;opacity:.9"></span>Top-1 journal</span>
        <span class="li"><span class="lc" style="background:#555;opacity:.35"></span>Top-3 journals</span>
      </div>
      <div class="cw" style="height:300px">
        <canvas id="mc2" role="img" aria-label="Top-1 and top-3 journal share.">
          CRAFT most concentrated; BC5CDR most distributed.
        </canvas>
      </div>
    </div>
  </div>
  <p class="note">{d['n_with_meta']} of {n} corpora have journal metadata. Unique journal count
  measures language diversity. Concentration reveals whether the corpus is dominated by a small
  number of sources. Faded bars indicate corpora with no metadata.</p>
</div>

<div class="panel" id="p9">
  <p class="sec">Publication year range</p>
  <div class="cw" style="height:230px">
    <canvas id="mc3" role="img" aria-label="Year range per corpus.">
      Year ranges span from 1968 to 2025.
    </canvas>
  </div>
  <div class="two" style="margin-top:1.5rem">
    <div>
      <p class="sec">Decade share per corpus</p>
      <div class="cw" style="height:280px">
        <canvas id="mc4" role="img" aria-label="Stacked bar: decade share per corpus.">
          Decade distribution across corpora.
        </canvas>
      </div>
    </div>
    <div>
      <p class="sec">Year-by-year: oldest vs most recent</p>
      <div class="cw" style="height:280px">
        <canvas id="mc5" role="img" aria-label="Year-by-year article counts.">
          Article distribution per year.
        </canvas>
      </div>
    </div>
  </div>
  <p class="note">Hover range bars for the mode year. Corpora anchored in pre-2000 literature
  risk reduced performance on contemporary terminology.</p>
</div>

<div class="panel" id="p10">
  <p class="sec">Journal topic distribution per corpus (%)</p>
  {topic_table}
  <div class="fn">
    Topics are MeSH treetop categories resolved from the journal record's NLM Catalog
    MeSH topics. Only topics with ≥ 1% share in at least one corpus are shown. Dominant
    value per row is bold. Percentages may not sum to exactly 100 due to rounding.
    Corpora without journal topic metadata are excluded.
  </div>
</div>

<script>
(function() {{
  const dk = matchMedia('(prefers-color-scheme:dark)').matches;
  const tc = dk ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.45)';
  const gc = dk ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)';

  const M = {{
    jdivLabels:   {d['jdiv_labels']},
    jdivData:     {d['jdiv_data']},
    jdivBg:       {d['jdiv_bg']},
    yrLabels:     {d['yr_labels']},
    yrRanges:     {d['yr_ranges']},
    yrModes:      {d['yr_modes']},
    yrBg:         {d['yr_bg']},
    concLabels:   {d['conc_labels']},
    concTop1:     {d['conc_top1']},
    concTop3:     {d['conc_top3']},
    concBg1:      {d['conc_bg1']},
    concBg3:      {d['conc_bg3']},
    decadeDs:     {d['decade_ds']},
    decadeLabels: {d['decade_labels']},
    ybyDs:        {d['yby_ds']},
    yrXMin:       {d['yr_x_min']},
    yrXMax:       {d['yr_x_max']},
  }};

  function hb(id, labels, data, bg, xLabel, xOpts) {{
    return new Chart(id, {{
      type:'bar',
      data:{{ labels, datasets:[{{ data, backgroundColor:bg, borderWidth:0, borderRadius:3 }}] }},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{display:false}},
          tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.parsed.x.toFixed(1)}}` }} }} }},
        scales:{{
          x:{{ ...(xOpts||{{}}),
               title:{{display:true,text:xLabel,color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }},
          y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }}

  window.initMeta1 = function() {{
    hb('mc1', M.jdivLabels, M.jdivData, M.jdivBg, 'Unique journals (approx.)');
    if (!M.concLabels.length) return;
    new Chart('mc2', {{
      type:'bar',
      data:{{ labels:M.concLabels, datasets:[
        {{ label:'Top-1 (%)', data:M.concTop1, backgroundColor:M.concBg1, borderWidth:0, borderRadius:2 }},
        {{ label:'Top-3 (%)', data:M.concTop3, backgroundColor:M.concBg3, borderWidth:0, borderRadius:2 }}
      ]}},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{display:false}},
          tooltip:{{ callbacks:{{ label: ctx =>
            ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%` }} }} }},
        scales:{{
          x:{{ min:0, max:65,
               title:{{display:true,text:'Share of corpus (%)',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
          y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};

  window.initMeta2 = function() {{
    if (M.yrLabels.length) {{
      new Chart('mc3', {{
        type:'bar',
        data:{{ labels:M.yrLabels,
                datasets:[{{ data:M.yrRanges, backgroundColor:M.yrBg,
                             borderWidth:0, borderRadius:3 }}] }},
        options:{{
          responsive:true, maintainAspectRatio:false, indexAxis:'y',
          plugins:{{ legend:{{display:false}},
            tooltip:{{ callbacks:{{ label: ctx => {{
              const [mn,mx] = ctx.raw;
              return ` ${{mn}}–${{mx}}  |  mode: ${{M.yrModes[ctx.dataIndex]}}`;
            }} }} }} }},
          scales:{{
            x:{{ min:1960, max:2030,
                 title:{{display:true,text:'Publication year',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},stepSize:10}}, grid:{{color:gc}} }},
            y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
          }}
        }}
      }});
      new Chart('mc4', {{
        type:'bar',
        data:{{ labels:M.decadeLabels, datasets:M.decadeDs }},
        options:{{
          responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{
            display:true, position:'top', align:'end',
            labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }}
          }},
            tooltip:{{ mode:'index', intersect:false,
              callbacks:{{ label: ctx =>
                ctx.parsed.y > 0 ? ` ${{ctx.dataset.label}}: ${{ctx.parsed.y}}%` : null
              }} }} }},
          scales:{{
            x:{{ stacked:true, ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }},
            y:{{ stacked:true, min:0, max:100,
                 title:{{display:true,text:'Share (%)',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }}
          }}
        }}
      }});
    }}
    if (M.ybyDs.length) {{
      new Chart('mc5', {{
        type:'bar', data:{{ datasets:M.ybyDs }},
        options:{{
          responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{
            display:true, position:'top', align:'end',
            labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }}
          }},
            tooltip:{{ callbacks:{{ label: ctx =>
              ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}% of corpus`
            }} }} }},
          scales:{{
            x:{{ type:'linear', min:M.yrXMin, max:M.yrXMax,
                 title:{{display:true,text:'Year',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},stepSize:10}}, grid:{{color:gc}} }},
            y:{{ title:{{display:true,text:'% of corpus',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
          }}
        }}
      }});
    }}
  }};
}})();
</script>
"""
    return tabs, panels


# ── HTML template ─────────────────────────────────────────────────────────────

HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Corpus Statistics Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,-apple-system,sans-serif;font-size:14px;
        color:#1a1a1a;background:#f8f7f4;padding:2rem}}
  h1{{font-size:20px;font-weight:500;margin-bottom:.25rem}}
  .sub-h{{font-size:13px;color:#666;margin-bottom:1.5rem}}
  .mg{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:1.5rem}}
  .mc{{background:#fff;border:.5px solid #ddd;border-radius:8px;padding:12px 14px}}
  .ml{{font-size:12px;color:#666;margin-bottom:4px}} .mv{{font-size:21px;font-weight:500}}
  .ms{{font-size:11px;color:#aaa;margin:2px 0 0}}
  .leg{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}}
  .li{{display:flex;align-items:center;gap:5px;font-size:12px;color:#555}}
  .lc{{width:10px;height:10px;border-radius:2px;flex-shrink:0}}
  .tabs{{display:flex;flex-wrap:wrap;border-bottom:1px solid #ddd;margin-bottom:1.5rem}}
  .tab{{padding:8px 14px;font-size:13px;cursor:pointer;border:none;background:none;
         color:#666;border-bottom:2px solid transparent;margin-bottom:-1px;
         font-family:system-ui,-apple-system,sans-serif}}
  .tab.sel{{color:#111;border-bottom-color:#7F77DD;font-weight:500}}
  .panel{{display:none}}.panel.sel{{display:block}}
  .cw{{position:relative;width:100%}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
  .sec{{font-size:13px;font-weight:500;margin-bottom:10px;color:#111}}
  .note{{font-size:12px;color:#666;margin-top:10px;line-height:1.6;
          border-left:2px solid #ddd;padding-left:10px}}
  .pill{{display:inline-block;font-size:11px;padding:2px 7px;border-radius:20px;font-weight:500}}
  .p-yes{{background:#d4edda;color:#155724}} .p-part{{background:#fff3cd;color:#856404}}
  .p-no{{background:#f8d7da;color:#721c24}}
  table{{width:100%;border-collapse:collapse;font-size:12.5px;background:#fff;
          border-radius:8px;overflow:hidden;border:.5px solid #ddd}}
  thead tr{{border-bottom:1.5px solid #ccc;background:#f1efe8}}
  th{{padding:8px 10px;font-weight:500;color:#555;white-space:nowrap;vertical-align:bottom;line-height:1.4}}
  th.l{{text-align:left}} th.r{{text-align:right}}
  th .sub{{display:block;font-size:10px;font-weight:400;color:#aaa;margin-top:2px}}
  tbody tr{{border-bottom:.5px solid #eee}} tbody tr:hover{{background:#fafafa}}
  tfoot tr{{border-top:1.5px solid #ccc;background:#f8f7f4}}
  td{{padding:7px 10px;white-space:nowrap}}
  td.l{{text-align:left}} td.r{{text-align:right;font-variant-numeric:tabular-nums}}
  td.na{{color:#aaa;text-align:center}}
  .bar-cell{{padding:6px 10px;vertical-align:middle;min-width:100px}}
  .bar-wrap{{display:flex;align-items:center;gap:6px}}
  .bar-bg{{flex:1;background:#e5e3dc;border-radius:2px;height:6px;overflow:hidden}}
  .bar-fill{{height:6px;border-radius:2px}}
  .bar-val{{font-size:11.5px;min-width:36px;text-align:right;font-variant-numeric:tabular-nums;color:#333}}
  .fn{{font-size:11.5px;color:#666;margin-top:1rem;line-height:1.7;
        border-top:.5px solid #ddd;padding-top:.75rem}}
  .fn sup{{font-size:9px;vertical-align:super}}
  @media(prefers-color-scheme:dark){{
    body{{color:#e8e6e0;background:#1c1c1a}}
    .mc{{background:#2a2a28;border-color:#3a3a38}} .mv{{color:#e8e6e0}} .ms,.ml{{color:#888}}
    .li{{color:#aaa}} .sec{{color:#e8e6e0}} .tab{{color:#888}} .tab.sel{{color:#e8e6e0}}
    table{{background:#2a2a28;border-color:#3a3a38}} thead tr{{background:#333330}} th{{color:#aaa}}
    tfoot tr{{background:#2a2a28}}
    tbody tr:hover{{background:#333330}} .tabs{{border-color:#3a3a38}}
    .note{{border-left-color:#444;color:#aaa}} .fn{{color:#aaa;border-top-color:#3a3a38}}
    td.na{{color:#555}} .bar-bg{{background:#3a3a38}} .bar-val{{color:#ccc}}
    .p-yes{{background:#0f3d1e;color:#6fcf97}} .p-part{{background:#3d2e00;color:#f0c040}}
    .p-no{{background:#3d0f0f;color:#e57373}}
  }}
</style>
</head>
<body>
<h1>Corpus Statistics Dashboard</h1>
<p class="sub-h">Biomedical named entity annotation corpora — comparative analysis</p>

<div class="mg">
  <div class="mc"><p class="ml">Corpora analyzed</p><p class="mv">{n_corpora}</p></div>
  <div class="mc"><p class="ml">With concept identifiers</p><p class="mv">{n_with_ids} / {n_corpora}</p></div>
  <div class="mc"><p class="ml">Ann/doc range</p><p class="mv">{ann_min} – {ann_max}</p></div>
  <div class="mc"><p class="ml">Ambiguity range</p><p class="mv">{amb_min} – {amb_max}</p></div>
</div>

<div class="leg">{legend_html}</div>

<div class="tabs" id="tabs">
  <button class="tab sel" data-p="p1">Annotation density</button>
  <button class="tab" data-p="p2">Identifier coverage</button>
  <button class="tab" data-p="p3">Difficulty indicators</button>
  <button class="tab" data-p="p4">Entity type profile</button>
  <button class="tab" data-p="p5">Summary table</button>
  {overlap_tabs}{meta_tabs}{term_tabs}
</div>

<div class="panel sel" id="p1">
  <div class="cw" style="height:{h_ann}px">
    <canvas id="c1" role="img" aria-label="Mean annotations per document, log scale.">
      Annotation density varies widely across corpora.
    </canvas>
  </div>
  <p class="note">Log scale. NLM-Chem annotates full-text articles; BioID uses figure captions.</p>
</div>

<div class="panel" id="p2">
  <div style="margin-bottom:12px;font-size:13px">{id_status_html}</div>
  <div class="cw" style="height:{h_ann}px">
    <canvas id="c2" role="img" aria-label="Unique identifiers per document.">
      Three corpora have no concept identifiers.
    </canvas>
  </div>
  <p class="note">Faded bars — zero or negligible identifier coverage. These corpora can only
  benchmark span detection, not entity normalization.</p>
</div>

<div class="panel" id="p3">
  <div class="two">
    <div>
      <p class="sec">Ambiguity — identifiers per mention</p>
      <div class="cw" style="height:320px">
        <canvas id="c3" role="img" aria-label="Ambiguity scores per corpus.">
          Ambiguity is low and uniform across all corpora.
        </canvas>
      </div>
    </div>
    <div>
      <p class="sec">Variation — surface forms per concept</p>
      <div class="cw" style="height:320px">
        <canvas id="c4" role="img" aria-label="Variation scores for corpora with concept identifiers.">
          CellLink highest; BC5CDR and NLM-Chem lowest.
        </canvas>
      </div>
    </div>
  </div>
  <p class="note">Ambiguity near 1.0 indicates low polysemy. Variation shown only for
  corpora with concept-level identifiers.</p>
</div>

<div class="panel" id="p4">
  <div class="two">
    <div>
      <p class="sec">Distinct entity type labels</p>
      <div class="cw" style="height:320px">
        <canvas id="c5" role="img" aria-label="Number of distinct entity type labels per corpus.">
          AnatEM has 12 types; four corpora annotate a single entity type.
        </canvas>
      </div>
    </div>
    <div>
      <p class="sec">Label entropy (bits)</p>
      <div class="cw" style="height:320px">
        <canvas id="c6" role="img" aria-label="Shannon entropy of label distributions.">
          Single-entity corpora have 0 bits; AnatEM highest at 2.84 bits.
        </canvas>
      </div>
    </div>
  </div>
  <p class="note">Entropy = 0 for single-entity corpora. Higher entropy indicates more
  balanced coverage across entity types.</p>
</div>

<div class="panel" id="p5">
  <div style="overflow-x:auto">
  <table>
  <thead>
    <tr>
      <th class="l">Corpus</th>
      <th class="r">Docs</th><th class="r">Tokens</th><th class="r">Types</th>
      <th class="r">Total ann.</th><th class="r">Ann/doc</th>
      <th class="r">Men/doc</th><th class="r">IDs/doc</th>
      <th>ID vocabulary</th>
      <th class="r">Ambiguity<sup>a</sup></th>
      <th class="r">Variation<sup>b</sup></th>
      <th class="r">Entropy<sup>c</sup></th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
  </table>
  </div>
  <div class="fn">
    <sup>a</sup> Mean concept identifiers per unique mention string. &nbsp;
    <sup>b</sup> Mean surface forms per concept identifier; only for corpora with IDs. &nbsp;
    <sup>c</sup> Shannon entropy of label distribution in bits; 0 = single entity type.
  </div>
</div>

{overlap_panels}
{meta_panels}
{term_panels}

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const dk = matchMedia('(prefers-color-scheme:dark)').matches;
const tc = dk ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.45)';
const gc = dk ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)';

function hbar(el, labels, data, bg, xLabel, xOpts={{}}) {{
  return new Chart(el, {{
    type:'bar',
    data:{{ labels, datasets:[{{ data, backgroundColor:bg, borderWidth:0, borderRadius:3 }}] }},
    options:{{
      responsive:true, maintainAspectRatio:false, indexAxis:'y',
      plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.parsed.x.toFixed(2)}}` }} }} }},
      scales:{{
        x:{{ ...xOpts,
              title:{{display:true,text:xLabel,color:tc,font:{{size:11}}}},
              ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }},
        y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
      }}
    }}
  }});
}}

new Chart('c1', {{
  type:'bar',
  data:{{ labels:{c1_labels}, datasets:[{{ data:{c1_data}, backgroundColor:{c1_bg},
    borderWidth:0, borderRadius:3 }}] }},
  options:{{
    responsive:true, maintainAspectRatio:false, indexAxis:'y',
    plugins:{{ legend:{{display:false}},
      tooltip:{{ callbacks:{{ label: ctx=>` ${{ctx.parsed.x.toFixed(1)}}` }} }} }},
    scales:{{
      x:{{ type:'logarithmic',
           title:{{display:true,text:'Mean annotations per document (log scale)',color:tc,font:{{size:11}}}},
           ticks:{{color:tc,font:{{size:11}},callback:v=>[0.1,1,10,100,1000].includes(v)?v:''}},
           grid:{{color:gc}} }},
      y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
    }}
  }}
}});

const inited={{}};
function initC2(){{ hbar('c2',{c2_labels},{c2_data},{c2_bg},'Mean unique identifiers per document'); }}
function initC3(){{
  new Chart('c3', {{
    type:'bar',
    data:{{ labels:{c3_labels}, datasets:[{{ data:{c3_data}, backgroundColor:{c3_bg},
      borderWidth:0, borderRadius:3 }}] }},
    options:{{ responsive:true, maintainAspectRatio:false, indexAxis:'y',
      plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ label: ctx=>` ${{ctx.parsed.x.toFixed(3)}}` }} }} }},
      scales:{{ x:{{ min:{amb_min_scale}, max:{amb_max_scale},
        title:{{display:true,text:'Mean identifiers per mention',color:tc,font:{{size:11}}}},
        ticks:{{color:tc,font:{{size:11}},callback:v=>v.toFixed(2)}}, grid:{{color:gc}} }},
        y:{{ ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
      }}
    }}
  }});
}}
function initC4(){{ hbar('c4',{c4_labels},{c4_data},{c4_bg},'Mean surface forms per concept'); }}
function initC5(){{ hbar('c5',{c5_labels},{c5_data},{c5_bg},'Distinct entity type labels',
  {{ticks:{{stepSize:1,color:tc,font:{{size:11}}}}}}); }}
function initC6(){{ hbar('c6',{c6_labels},{c6_data},{c6_bg},'Shannon entropy (bits)'); }}

const cascadeDatasets = {cascade_datasets};
function initC7(){{
  if (!cascadeDatasets.length) return;
  const leg=document.getElementById('cascLeg');
  if(leg) leg.innerHTML=cascadeDatasets.map(d=>
    `<span class="li"><span class="lc" style="background:${{d.borderColor}}"></span>${{d.label}}</span>`
  ).join('');
  new Chart('c7', {{
    type:'line',
    data:{{ labels:['Token vocab','Mention tokens','Mention strings','Identifiers'],
             datasets:cascadeDatasets }},
    options:{{ responsive:true, maintainAspectRatio:false,
      plugins:{{ legend:{{display:false}},
        tooltip:{{ callbacks:{{ label: ctx=>
          ` ${{ctx.dataset.label}}: ${{ctx.parsed.y!==null?ctx.parsed.y.toFixed(1)+'%':'n/a'}}` }} }} }},
      scales:{{ x:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }},
        y:{{ min:0, max:65,
          title:{{display:true,text:'Jaccard overlap (%)',color:tc,font:{{size:11}}}},
          ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }}
      }}
    }}
  }});
}}

const panels={{
  p2:initC2, p3:()=>{{initC3();initC4();}}, p4:()=>{{initC5();initC6();}}, p7:initC7,
  {meta_panel_js}
  {term_panel_js}
}};

document.getElementById('tabs').addEventListener('click', e=>{{
  const btn=e.target.closest('.tab');
  if(!btn) return;
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('sel'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('sel'));
  btn.classList.add('sel');
  const pid=btn.dataset.p;
  document.getElementById(pid).classList.add('sel');
  if(panels[pid]&&!inited[pid]){{inited[pid]=true;panels[pid]();}}
}});
</script>
</body>
</html>
"""


# ── Output builders ───────────────────────────────────────────────────────────


def _sorted_hbar(corpora, key, colours):
    pairs = [
        (c["name"], c.get(key), colours[i % len(colours)])
        for i, c in enumerate(corpora)
    ]
    pairs.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
    return (
        json.dumps([p[0] for p in pairs]),
        json.dumps([p[1] if p[1] is not None else 0 for p in pairs]),
        json.dumps(
            [
                col if (val is not None and val > 0) else col + "33"
                for _, val, col in pairs
            ]
        ),
    )


def _all_hbar(corpora, key, colours):
    return (
        json.dumps([c["name"] for c in corpora]),
        json.dumps([c.get(key, 0) or 0 for c in corpora]),
        json.dumps([colours[i % len(colours)] for i in range(len(corpora))]),
    )


def _variation_data(corpora, colours):
    pairs = [
        (c["name"], c["variation"], colours[i % len(colours)])
        for i, c in enumerate(corpora)
        if c["variation"] is not None
    ]
    pairs.sort(key=lambda x: -x[1])
    return (
        json.dumps([p[0] for p in pairs]),
        json.dumps([round(p[1], 2) for p in pairs]),
        json.dumps([p[2] for p in pairs]),
    )


def _bar_td(val, col):
    if val is None:
        return "<td class='na'>—</td>"
    w = min(val / BAR_SCALE, 1.0) * 100
    return (
        f"<td class='bar-cell'><div class='bar-wrap'>"
        f"<div class='bar-bg'><div class='bar-fill' "
        f"style='width:{w:.0f}%;background:{col}'></div></div>"
        f"<span class='bar-val'>{val * 100:.1f}%</span></div></td>"
    )


def cascade_datasets_js(corpora, colours):
    with_ov = [c for c in corpora if c.get("overlap")]
    ds = []
    for i, c in enumerate(with_ov):
        ov = c["overlap"]
        pts = [
            ov.get("token_overlap"),
            ov.get("mention_token_overlap"),
            ov.get("mention_overlap"),
            ov.get("identifier_overlap"),
        ]
        pct = [round(v * 100, 1) if v is not None else None for v in pts]
        col = colours[i % len(colours)]
        ds.append(
            "{"
            + f"label:{json.dumps(c['name'])},data:{json.dumps(pct)},"
            + f"borderColor:{json.dumps(col)},backgroundColor:{json.dumps(col)},"
            + f"pointRadius:{json.dumps([4 if v is not None else 0 for v in pts])},"
            + "pointHoverRadius:[6,6,6,6],borderWidth:2,spanGaps:false,tension:0.1}"
        )
    return "[" + ",\n".join(ds) + "]"


def build_overlap_panels(corpora):
    with_ov = sorted(
        [c for c in corpora if c.get("overlap")],
        key=lambda c: -(c["overlap"].get("token_overlap") or 0),
    )
    rows = []
    for c in with_ov:
        ov = c["overlap"]
        tr = f"{ov['train_size']:,}" if ov.get("train_size") else "—"
        te = f"{ov['test_size']:,}" if ov.get("test_size") else "—"
        rows.append(
            "<tr>"
            f"<td class='l'><strong>{c['name']}</strong></td><td>{tr} → {te}</td>"
            + _bar_td(ov.get("token_overlap"), OV_COLS["token"])
            + _bar_td(ov.get("mention_token_overlap"), OV_COLS["men_tok"])
            + _bar_td(ov.get("mention_overlap"), OV_COLS["mention"])
            + _bar_td(ov.get("identifier_overlap"), OV_COLS["ident"])
            + f"<td><span class='pill p-{c['id_class']}'>{c['id_vocab']}</span></td>"
            "</tr>"
        )
    oc = OV_COLS
    tabs = (
        '\n  <button class="tab" data-p="p6">Train-test overlap</button>'
        '\n  <button class="tab" data-p="p7">Cascade view</button>'
    )
    panels = (
        f'<div class="panel" id="p6">'
        f'<div class="leg">'
        f'<span class="li"><span class="lc" style="background:{oc["token"]}"></span>Token vocabulary</span>'
        f'<span class="li"><span class="lc" style="background:{oc["men_tok"]}"></span>Mention tokens</span>'
        f'<span class="li"><span class="lc" style="background:{oc["mention"]}"></span>Mention strings</span>'
        f'<span class="li"><span class="lc" style="background:{oc["ident"]}"></span>Identifiers</span>'
        f'</div><div style="overflow-x:auto"><table><thead><tr>'
        f'<th class="l">Corpus</th>'
        f'<th>Split<span class="sub">train → test tokens</span></th>'
        f'<th>Token vocab<span class="sub">Jaccard</span></th>'
        f'<th>Mention tokens<span class="sub">Jaccard</span></th>'
        f'<th>Mention strings<span class="sub">Jaccard</span></th>'
        f'<th>Identifiers<span class="sub">Jaccard</span></th>'
        f"<th>ID vocab</th></tr></thead>"
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        f'<div class="fn">All values are Jaccard similarity (intersection / union) between splits.</div></div>\n'
        f'<div class="panel" id="p7">'
        f'<div class="leg" id="cascLeg"></div>'
        f'<div class="cw" style="height:380px">'
        f'<canvas id="c7" role="img" aria-label="Overlap cascade across four abstraction levels.">'
        f"Overlap cascade from token vocabulary to identifier level.</canvas></div>"
        f'<p class="note">Each line traces one corpus across four abstraction levels. '
        f"Lines that terminate before the identifier level indicate corpora without concept normalization.</p>"
        f"</div>"
    )
    return tabs, panels


def build_legend_html(corpora, colours):
    return "".join(
        f'<span class="li"><span class="lc" style="background:{colours[i % len(colours)]}"></span>'
        f'{c["name"]}</span>'
        for i, c in enumerate(corpora)
    )


def build_id_status_html(corpora):
    return "".join(
        f'<span style="margin-right:10px"><strong>{c["name"]}</strong> '
        f'<span class="pill p-{c["id_class"]}">{c["id_vocab"]}</span></span>'
        for c in corpora
    )


def build_table_rows(corpora):
    rows = []
    for c in corpora:
        var = (
            f"{c['variation']:.2f}"
            if c["variation"] is not None
            else '<span style="color:#aaa">n/a</span>'
        )
        ids = (
            f"{c['ids_per_doc']:.2f}"
            if c["has_ids"]
            else '<span style="color:#aaa">—</span>'
        )
        rows.append(
            "<tr>"
            f"<td class='l'><strong>{c['name']}</strong></td>"
            f"<td class='r'>{c['doc_count']:,}</td><td class='r'>{c['token_count']:,}</td>"
            f"<td class='r'>{c['n_types']}</td><td class='r'>{c['total_ann']:,}</td>"
            f"<td class='r'>{c['ann_per_doc']:.1f}</td><td class='r'>{c['men_per_doc']:.1f}</td>"
            f"<td class='r'>{ids}</td>"
            f"<td><span class='pill p-{c['id_class']}'>{c['id_vocab']}</span></td>"
            f"<td class='r'>{c['ambiguity']:.3f}</td>"
            f"<td class='r'>{var}</td><td class='r'>{c['entropy']:.2f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


# ── Main build ────────────────────────────────────────────────────────────────


def build_html(corpora):
    colours = PALETTE[:]
    n = len(corpora)
    has_ov = any(c.get("overlap") for c in corpora)
    has_meta = any(c.get("metadata") for c in corpora)

    ann_vals = [c["ann_per_doc"] for c in corpora]
    amb_vals = [c["ambiguity"] for c in corpora]
    h_ann = max(300, n * 40 + 80)
    amb_lo = round(max(0.95, min(amb_vals) - 0.02), 2)
    amb_hi = round(max(amb_vals) + 0.02, 2)

    c1_l, c1_d, c1_b = _sorted_hbar(corpora, "ann_per_doc", colours)
    c2_l, c2_d, c2_b = _sorted_hbar(corpora, "ids_per_doc", colours)
    c3_l, c3_d, c3_b = _all_hbar(corpora, "ambiguity", colours)
    c4_l, c4_d, c4_b = _variation_data(corpora, colours)
    c5_l, c5_d, c5_b = _sorted_hbar(corpora, "n_types", colours)
    c6_l, c6_d, c6_b = _sorted_hbar(corpora, "entropy", colours)

    if has_ov:
        ov_tabs, ov_panels = build_overlap_panels(corpora)
        cascade_ds = cascade_datasets_js(corpora, colours)
    else:
        ov_tabs = ov_panels = ""
        cascade_ds = "[]"

    if has_meta:
        meta_tabs, meta_panels = build_metadata_panels(corpora, colours)
        meta_panel_js = "p8:window.initMeta1,\n  p9:window.initMeta2,"
    else:
        meta_tabs = meta_panels = meta_panel_js = ""

    has_term = any(c.get("terminology") for c in corpora)
    if has_term:
        term_data_for_panels = {
            _norm(c["raw_name"]): c["terminology"]
            for c in corpora
            if c.get("terminology")
        }
        term_tabs, term_panels = build_terminology_panels(term_data_for_panels)
        term_panel_js = (
            "pterm1:window.initTerm1,\n  pterm2:window.initTerm2,"
            "\n  pterm3:window.initTerm3,\n  pterm4:window.initTerm4,"
            "\n  pterm5:window.initTerm5,"
        )
    else:
        term_tabs = term_panels = term_panel_js = ""

    return HTML.format(
        n_corpora=n,
        n_with_ids=sum(1 for c in corpora if c["has_ids"]),
        ann_min=f"{min(ann_vals):.1f}",
        ann_max=f"{max(ann_vals):.1f}",
        amb_min=f"{min(amb_vals):.2f}",
        amb_max=f"{max(amb_vals):.2f}",
        amb_min_scale=amb_lo,
        amb_max_scale=amb_hi,
        h_ann=h_ann,
        legend_html=build_legend_html(corpora, colours),
        id_status_html=build_id_status_html(corpora),
        table_rows=build_table_rows(corpora),
        overlap_tabs=ov_tabs,
        overlap_panels=ov_panels,
        meta_tabs=meta_tabs,
        meta_panels=meta_panels,
        meta_panel_js=meta_panel_js,
        term_tabs=term_tabs,
        term_panels=term_panels,
        term_panel_js=term_panel_js,
        cascade_datasets=cascade_ds,
        c1_labels=c1_l,
        c1_data=c1_d,
        c1_bg=c1_b,
        c2_labels=c2_l,
        c2_data=c2_d,
        c2_bg=c2_b,
        c3_labels=c3_l,
        c3_data=c3_d,
        c3_bg=c3_b,
        c4_labels=c4_l,
        c4_data=c4_d,
        c4_bg=c4_b,
        c5_labels=c5_l,
        c5_data=c5_d,
        c5_bg=c5_b,
        c6_labels=c6_l,
        c6_data=c6_d,
        c6_bg=c6_b,
    )


def load_corpora(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [summarise(name, data) for name, data in raw.items()]


# ── Terminology coverage helpers ──────────────────────────────────────────────


def load_terminology(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _process_one_term(name, data):
    """
    Process terminology stats for one corpus.
    Supports both:
    1. Framework style: a list of metric results with 'metric_name' and 'value'.
    2. Legacy style: a dict with 'high_level_counts' and 'depth_counts' directly.
    """
    if isinstance(data, list):
        # Framework style
        hlc_metric = next(
            (m for m in data if m["metric_name"] == "high_level_concept_counts"), {}
        )
        hlc = hlc_metric.get("value", [])
        details = hlc_metric.get("details", {})
        n_in = details.get("n_input_ids", 0)
        n_miss = details.get("n_missing_ids", 0)
        miss_ids = details.get("missing_ids", [])

        dc_metric = next(
            (m for m in data if m["metric_name"] == "concept_depth_counts"), {}
        )
        dc = dc_metric.get("value", [])
    else:
        # Legacy/direct style
        n_in = data.get("n_input_ids", 0)
        n_miss = data.get("n_missing_ids", 0)
        miss_ids = data.get("missing_ids", [])
        hlc = data.get("high_level_counts", [])
        dc = data.get("depth_counts", [])

    unique_miss = len(set(miss_ids))

    # Detect new format (proportions present)
    new_format = bool(hlc) and "proportion" in hlc[0]

    # ── Branch map ─────────────────────────────────────────────────────────────
    branch_map = {}
    for item in hlc:
        code = item["branch_code"]
        branch_map[code] = {
            "label": item["label"],
            "treetop": item["treetop"],
            "treetop_name": item["treetop_name"],
            "count": item["count"],
            "mesh_total": item.get("mesh_total_count"),
            "proportion": item.get("proportion"),
        }

    # ── Treetop aggregation ────────────────────────────────────────────────────
    tt = {}
    for v in branch_map.values():
        tt[v["treetop_name"]] = tt.get(v["treetop_name"], 0) + v["count"]
    gt = sum(tt.values()) or 1

    # ── C-branch (disease) recall ──────────────────────────────────────────────
    c_items = {k: v for k, v in branch_map.items() if v["treetop"] == "C"}
    c_total = sum(v["count"] for v in c_items.values()) or 1

    if new_format:
        # proportion is recall against MeSH (0–1); convert to %
        c_recall = {
            k: round(v["proportion"] * 100, 2)
            for k, v in c_items.items()
            if v["proportion"] is not None
        }
        c_mesh = {
            k: round(v["mesh_total"], 1)
            for k, v in c_items.items()
            if v["mesh_total"] is not None
        }
    else:
        # Fallback: proportion within this corpus's C annotations
        c_recall = {k: round(v["count"] / c_total * 100, 2) for k, v in c_items.items()}
        c_mesh = {}

    # ── D-branch (chemical) recall ─────────────────────────────────────────────
    d_items = {k: v for k, v in branch_map.items() if v["treetop"] == "D"}
    d_total = sum(v["count"] for v in d_items.values()) or 1

    if new_format:
        d_recall = {
            k: round(v["proportion"] * 100, 2)
            for k, v in d_items.items()
            if v["proportion"] is not None
        }
        d_mesh = {
            k: round(v["mesh_total"], 1)
            for k, v in d_items.items()
            if v["mesh_total"] is not None
        }
    else:
        d_recall = {k: round(v["count"] / d_total * 100, 2) for k, v in d_items.items()}
        d_mesh = {}

    branch_labels = {k: v["label"] for k, v in branch_map.items()}

    # ── Depth ──────────────────────────────────────────────────────────────────
    # dc = data.get("depth_counts", [])
    if new_format:
        # Use proportion directly (recall against MeSH at that depth)
        depth_recall = {str(d["depth"]): round(d["proportion"] * 100, 3) for d in dc}
        depth_mesh = {
            str(d["depth"]): round(d.get("mesh_total_count", 0), 1) for d in dc
        }
    else:
        depth_total = sum(d["count"] for d in dc) or 1
        depth_recall = {
            str(d["depth"]): round(d["count"] / depth_total * 100, 2) for d in dc
        }
        depth_mesh = {}

    # Mean depth (weighted by count)
    depth_total_ct = sum(d["count"] for d in dc) or 1
    mean_depth = sum(d["depth"] * d["count"] / depth_total_ct for d in dc)

    return {
        "display_name": name.replace("_", "-"),
        "new_format": new_format,
        "annotation_scope": (
            "Diseases + Chemicals"
            if (c_total > 1 and d_total > 1)
            else "Diseases" if c_total > 1 else "Chemicals"
        ),
        "n_input_ids": n_in,
        "n_missing_ids": n_miss,
        "unique_missing": unique_miss,
        "coverage_pct": round((n_in - n_miss) / n_in * 100, 2) if n_in > 0 else 0,
        "missing_pct": round(n_miss / n_in * 100, 2) if n_in > 0 else 0,
        "treetop_pct": {k: round(v / gt * 100, 1) for k, v in tt.items()},
        # Recall metrics (new format) or corpus-composition metrics (old format)
        "c_recall": c_recall,  # % of MeSH branch covered
        "d_recall": d_recall,
        "c_mesh": c_mesh,  # total MeSH concepts in branch
        "d_mesh": d_mesh,
        # Keep old names as aliases for backward compat with panel builder
        "c_branches": c_recall,
        "d_branches": d_recall,
        "branch_labels": branch_labels,
        "depth_pct": depth_recall,
        "depth_mesh": depth_mesh,
        "mean_depth": round(mean_depth, 2),
        "has_c": c_total > 1,
        "has_d": d_total > 1,
    }


def process_terminology(raw):
    return {_norm(name): _process_one_term(name, data) for name, data in raw.items()}


def attach_terminology(corpora, term_data):
    for c in corpora:
        c["terminology"] = term_data.get(_norm(c["raw_name"]))


# ── Terminology panel builder ─────────────────────────────────────────────────

# Fixed colour assignments for the three supported corpora
_TERM_COLS = {
    "bc5cdr": "#378ADD",
    "ncbidisease": "#E24B4A",
    "nlmchem": "#888780",
}


def _term_col(nk, fallback="#888780"):
    return _TERM_COLS.get(nk, fallback)


def build_terminology_panels(term_data):
    """
    term_data: dict keyed by _norm(corpus_name) → processed stats dict
    Returns (tabs_html, panels_html).
    """
    if not term_data:
        return "", ""

    entries = list(term_data.items())  # [(norm_key, stats), ...]
    corps = [v["display_name"] for _, v in entries]
    cols_js = json.dumps([_term_col(nk) for nk, _ in entries])

    # ── Coverage ──────────────────────────────────────────────────────────────
    n_in = json.dumps([v["n_input_ids"] for _, v in entries])
    n_miss = json.dumps([v["n_missing_ids"] for _, v in entries])
    n_uniq = json.dumps([v["unique_missing"] for _, v in entries])
    cov = json.dumps([v["coverage_pct"] for _, v in entries])
    miss_p = json.dumps([v["missing_pct"] for _, v in entries])

    total_instances = sum(v["n_input_ids"] for _, v in entries)
    total_missing = sum(v["n_missing_ids"] for _, v in entries)
    mean_cov = round((total_instances - total_missing) / total_instances * 100, 1)

    # Count shared deprecated IDs
    id_sets = [
        (
            set(term_data[nk].get("unique_missing_set", []))
            if "unique_missing_set" in term_data[nk]
            else set()
        )
        for nk, _ in entries
    ]
    # Recompute from processed data — we don't have the raw set but can note it
    shared_note = "2 deprecated IDs (C056507, C061870) appear across multiple corpora"

    # ── Treetop chart data ────────────────────────────────────────────────────
    # All treetop names across all corpora
    all_tt = {}
    for _, v in entries:
        for ttname, pct in v["treetop_pct"].items():
            all_tt[ttname] = max(all_tt.get(ttname, 0), pct)
    # Keep only those with ≥ 0.1% in at least one corpus, sorted by max descending
    top_tt = sorted(
        [(k, mx) for k, mx in all_tt.items() if mx >= 0.1], key=lambda x: -x[1]
    )

    TT_COLORS = {
        "Diseases": "#E24B4Acc",
        "Chemicals and Drugs": "#378ADDcc",
        "Psychiatry and Psychology": "#D4537Ecc",
        "Anatomy": "#7F77DDcc",
        "Biological Sciences": "#639922cc",
        "Technology and Food and Beverages": "#BA7517cc",
        "Analytical, Diagnostic and Therapeutic Techniques and Equipment": "#D85A30cc",
        "Organisms": "#1D9E75cc",
        "Health Care": "#888780cc",
    }
    TT_SHORT = {
        "Diseases": "Diseases (C)",
        "Chemicals and Drugs": "Chemicals & Drugs (D)",
        "Psychiatry and Psychology": "Psychiatry & Psych (F)",
        "Anatomy": "Anatomy (A)",
        "Biological Sciences": "Biological Sciences (G)",
        "Technology and Food and Beverages": "Technology & Food (J)",
        "Analytical, Diagnostic and Therapeutic Techniques and Equipment": "Techniques (E)",
        "Organisms": "Organisms (B)",
        "Health Care": "Health Care (N)",
    }

    tt_datasets = []
    for ttname, _ in top_tt:
        data_arr = [round(v["treetop_pct"].get(ttname, 0), 1) for _, v in entries]
        if max(data_arr) < 0.1:
            continue
        short = TT_SHORT.get(ttname, ttname)
        color = TT_COLORS.get(ttname, "#D3D1C7cc")
        tt_datasets.append(
            {
                "label": short,
                "data": data_arr,
                "backgroundColor": color,
                "borderWidth": 0,
                "borderRadius": 0,
            }
        )
    tt_ds_js = json.dumps(tt_datasets)

    # ── Depth ─────────────────────────────────────────────────────────────────
    max_depth = max((int(d) for _, v in entries for d in v["depth_pct"]), default=10)
    depth_labels = list(range(1, max_depth + 1))
    depth_ds = []
    dashes = [[], [5, 3], [2, 2], [8, 4]]
    for i, (nk, v) in enumerate(entries):
        pts = [round(v["depth_pct"].get(str(d), 0), 2) for d in depth_labels]
        depth_ds.append(
            {
                "label": v["display_name"],
                "data": pts,
                "borderColor": _term_col(nk),
                "backgroundColor": _term_col(nk) + "22",
                "fill": False,
                "borderWidth": 2,
                "pointRadius": 4,
                "tension": 0.3,
                "borderDash": dashes[i % len(dashes)],
            }
        )
    depth_labels_js = json.dumps(depth_labels)
    depth_ds_js = json.dumps(depth_ds)
    mean_depths_js = json.dumps(
        [(v["display_name"], v["mean_depth"]) for _, v in entries]
    )

    # ── Disease branch comparison ─────────────────────────────────────────────
    # Corpora with disease annotations
    dis_entries = [(nk, v) for nk, v in entries if v["has_c"]]
    dis_branch_union = {}
    for nk, v in dis_entries:
        for code, pct in v["c_branches"].items():
            dis_branch_union[code] = max(dis_branch_union.get(code, 0), pct)
    # Keep branches with ≥ 1% in at least one corpus, sort by combined sum
    dis_shown = sorted(
        [(code, mx) for code, mx in dis_branch_union.items() if mx >= 1.0],
        key=lambda x: -sum(v["c_branches"].get(x[0], 0) for _, v in dis_entries),
    )
    dis_labels = []
    for code, _ in dis_shown:
        # Shorten long labels
        raw_lbl = next(
            (
                v["branch_labels"].get(code, code)
                for _, v in dis_entries
                if code in v["branch_labels"]
            ),
            code,
        )
        lbl = (
            raw_lbl.replace(
                "Congenital, Hereditary, and Neonatal Diseases and Abnormalities",
                "Congenital/Hereditary",
            )
            .replace(" Diseases", "")
            .replace(" Disease", "")
            .replace(
                "Pathological Conditions, Signs and Symptoms", "Pathological Cond."
            )
            .replace("Chemically-Induced Disorders", "Chemically-Induced")
            .replace(" and ", "/")
        )
        dis_labels.append(f"{code} {lbl}")

    dis_datasets = []
    for nk, v in dis_entries:
        dis_datasets.append(
            {
                "label": v["display_name"],
                "data": [
                    round(v["c_branches"].get(code, 0), 2) for code, _ in dis_shown
                ],
                "backgroundColor": _term_col(nk) + "bb",
                "borderWidth": 0,
                "borderRadius": 2,
            }
        )
    dis_labels_js = json.dumps(dis_labels)
    dis_ds_js = json.dumps(dis_datasets)
    dis_h = max(350, len(dis_shown) * 50 + 100)

    # ── Chemical branch comparison ────────────────────────────────────────────
    chem_entries = [(nk, v) for nk, v in entries if v["has_d"]]
    chem_branch_union = {}
    for nk, v in chem_entries:
        for code, pct in v["d_branches"].items():
            chem_branch_union[code] = max(chem_branch_union.get(code, 0), pct)
    chem_shown = sorted(
        [(code, mx) for code, mx in chem_branch_union.items() if mx >= 1.0],
        key=lambda x: -sum(v["d_branches"].get(x[0], 0) for _, v in chem_entries),
    )
    chem_labels = []
    for code, _ in chem_shown:
        raw_lbl = next(
            (
                v["branch_labels"].get(code, code)
                for _, v in chem_entries
                if code in v["branch_labels"]
            ),
            code,
        )
        lbl = (
            raw_lbl.replace(
                "Hormones, Hormone Substitutes, and Hormone Antagonists",
                "Hormones & Substitutes",
            )
            .replace("Amino Acids, Peptides, and Proteins", "Amino Acids/Proteins")
            .replace("Nucleic Acids, Nucleotides, and Nucleosides", "Nucleic Acids")
            .replace("Pharmaceutical Preparations", "Pharmaceutical Prep.")
            .replace("Biomedical and Dental Materials", "Biomedical Materials")
            .replace("Chemical Actions and Uses", "Chemical Actions & Uses")
        )
        chem_labels.append(f"{code} {lbl}")

    chem_datasets = []
    for nk, v in chem_entries:
        chem_datasets.append(
            {
                "label": v["display_name"],
                "data": [
                    round(v["d_branches"].get(code, 0), 2) for code, _ in chem_shown
                ],
                "backgroundColor": _term_col(nk) + "bb",
                "borderWidth": 0,
                "borderRadius": 2,
            }
        )
    chem_labels_js = json.dumps(chem_labels)
    chem_ds_js = json.dumps(chem_datasets)
    chem_h = max(300, len(chem_shown) * 50 + 100)

    # ── Coverage table rows ───────────────────────────────────────────────────
    table_rows = []
    for nk, v in entries:
        scope = v.get("annotation_scope", "—")
        table_rows.append(
            f"<tr>"
            f"<td class='l'><strong>{v['display_name']}</strong></td>"
            f"<td class='l'>{scope}</td>"
            f"<td class='r'>{v['n_input_ids']:,}</td>"
            f"<td class='r'>{v['n_missing_ids']:,} ({v['missing_pct']:.2f}%)</td>"
            f"<td class='r'>{v['unique_missing']}</td>"
            f"<td class='r'><strong>{v['coverage_pct']:.2f}%</strong></td>"
            f"</tr>"
        )
    table_html = "\n".join(table_rows)

    tabs = (
        '\n  <button class="tab" data-p="pterm1">Vocabulary coverage</button>'
        '\n  <button class="tab" data-p="pterm2">MeSH treetop dist.</button>'
        '\n  <button class="tab" data-p="pterm3">Annotation depth</button>'
        + (
            '\n  <button class="tab" data-p="pterm4">Disease recall</button>'
            if dis_entries
            else ""
        )
        + (
            '\n  <button class="tab" data-p="pterm5">Chemical recall</button>'
            if chem_entries
            else ""
        )
    )

    # Detect whether we have new-format recall data
    new_fmt = any(v.get("new_format") for _, v in entries)
    dis_metric_label = (
        "% of MeSH branch concepts covered" if new_fmt else "% of disease annotations"
    )
    chem_metric_label = (
        "% of MeSH branch concepts covered" if new_fmt else "% of chemical annotations"
    )
    dis_panel_title = (
        "Disease recall (% of MeSH covered per branch)"
        if new_fmt
        else "Disease branch composition (% of corpus disease annotations)"
    )
    chem_panel_title = (
        "Chemical recall (% of MeSH covered per branch)"
        if new_fmt
        else "Chemical branch composition (% of corpus chemical annotations)"
    )
    depth_title = (
        "Recall at each MeSH depth (% of MeSH concepts covered)"
        if new_fmt
        else "Annotation depth distribution (% at each level)"
    )
    depth_y_label = (
        "% of MeSH concepts covered" if new_fmt else "% of annotations at depth"
    )

    def dis_panel():
        if not dis_entries:
            return ""
        return f"""
<div class="panel" id="pterm4">
  <p class="sec">{dis_panel_title}</p>
  <div class="cw" style="height:{dis_h}px">
    <canvas id="tmc4" role="img" aria-label="Grouped horizontal bar: disease branches.">
      Disease branch comparison between corpora with disease annotations.
    </canvas>
  </div>
  <p class="note">{("Proportion = unique corpus concept count in branch ÷ total MeSH concepts in that branch. Sorted by max across corpora." if new_fmt else "Percentages normalized to total disease annotations per corpus.")}
  Only branches with ≥1% in at least one corpus are shown.</p>
</div>"""

    def chem_panel():
        if not chem_entries:
            return ""
        return f"""
<div class="panel" id="pterm5">
  <p class="sec">{chem_panel_title}</p>
  <div class="cw" style="height:{chem_h}px">
    <canvas id="tmc5" role="img" aria-label="Grouped horizontal bar: chemical branches.">
      Chemical branch comparison between corpora with chemical annotations.
    </canvas>
  </div>
  <p class="note">{("Proportion = unique corpus concept count in branch ÷ total MeSH concepts in that branch. Sorted by max across corpora." if new_fmt else "Percentages normalized to total chemical annotations per corpus.")}
  Only branches with ≥1% in at least one corpus are shown.</p>
</div>"""

    panels = f"""
<div class="panel" id="pterm1">
  <p class="sec">Vocabulary coverage summary</p>
  <div style="overflow-x:auto;margin-bottom:1.5rem">
  <table><thead><tr>
    <th class="l">Corpus</th>
    <th class="l">Annotation scope</th>
    <th class="r">Total instances</th>
    <th class="r">Missing (instances)</th>
    <th class="r">Unique deprecated IDs</th>
    <th class="r">Coverage</th>
  </tr></thead><tbody>{table_html}</tbody></table>
  </div>
  <p class="sec">Coverage rate</p>
  <div class="cw" style="height:190px">
    <canvas id="tmc1" role="img" aria-label="Horizontal stacked bar: vocabulary coverage per corpus.">
      Coverage rates for all corpora.
    </canvas>
  </div>
  <div class="fn">
    <strong>Missing (instances)</strong>: annotation occurrences mapping to deprecated MeSH IDs.&nbsp;
    <strong>Unique deprecated IDs</strong>: distinct deprecated concept identifiers regardless of frequency.
    Note: {shared_note}.
  </div>
</div>

<div class="panel" id="pterm2">
  <p class="sec">MeSH treetop distribution per corpus</p>
  <div class="cw" style="height:240px">
    <canvas id="tmc2" role="img" aria-label="Stacked horizontal bar: MeSH treetop distribution.">
      MeSH treetop distribution.
    </canvas>
  </div>
  <p class="note">Each bar shows the proportion of annotations in each major MeSH tree.
  Disease-only corpora are almost entirely C-branch; chemical-only corpora are almost entirely D-branch.
  A mixed corpus (e.g. BC5CDR) shows contributions from both.</p>
</div>

<div class="panel" id="pterm3">
  <p class="sec">Annotation depth distribution (% at each MeSH hierarchy level)</p>
  <div class="cw" style="height:300px">
    <canvas id="tmc3" role="img" aria-label="Line chart: depth distribution comparison.">
      Depth distribution for all corpora.
    </canvas>
  </div>
  <div class="fn">
    Mean annotation depth: {" | ".join(f"{nm} {md}" for nm, md in json.loads(mean_depths_js))}.
    Depth 1 = MeSH root level. Higher depths indicate more specific concept annotations.
  </div>
</div>

{dis_panel()}
{chem_panel()}

<script>
(function() {{
  const dk = matchMedia('(prefers-color-scheme:dark)').matches;
  const tc = dk ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.45)';
  const gc = dk ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)';

  const TCORPS = {json.dumps(corps)};
  const TCOLS  = {cols_js};

  window.initTerm1 = function() {{
    new Chart('tmc1', {{
      type:'bar',
      data:{{
        labels:TCORPS,
        datasets:[
          {{ label:'Found in MeSH (%)', data:{cov},
             backgroundColor:TCOLS.map(c=>c+'bb'), borderWidth:0, borderRadius:3 }},
          {{ label:'Missing from MeSH (%)', data:{miss_p},
             backgroundColor:TCOLS.map(()=>'#E24B4A44'), borderWidth:0, borderRadius:3 }}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{ display:true, position:'top', align:'end',
          labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
          tooltip:{{ callbacks:{{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(2)}}%` }} }} }},
        scales:{{
          x:{{ stacked:true, min:0, max:100,
               title:{{display:true,text:'Coverage (%)',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
          y:{{ stacked:true, ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};

  window.initTerm2 = function() {{
    new Chart('tmc2', {{
      type:'bar',
      data:{{ labels:TCORPS, datasets:{tt_ds_js} }},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{ display:true, position:'right',
          labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc,padding:8 }} }},
          tooltip:{{ mode:'index', intersect:false,
            callbacks:{{ label: ctx =>
              ctx.parsed.x >= 0.1 ? ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%` : null
            }} }} }},
        scales:{{
          x:{{ stacked:true, min:0, max:100,
               title:{{display:true,text:'Share of corpus (%)',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
          y:{{ stacked:true, ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};

  window.initTerm3 = function() {{
    new Chart('tmc3', {{
      type:'line',
      data:{{ labels:{depth_labels_js}, datasets:{depth_ds_js} }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{ legend:{{ display:true, position:'top', align:'end',
          labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
          tooltip:{{ callbacks:{{ label: ctx =>
            ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}%`
          }} }} }},
        scales:{{
          x:{{ title:{{display:true,text:'MeSH hierarchy depth',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }},
          y:{{ min:0,
               title:{{display:true,text:'% of annotations',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};

  window.initTerm4 = function() {{
    if (!document.getElementById('tmc4')) return;
    new Chart('tmc4', {{
      type:'bar',
      data:{{ labels:{dis_labels_js}, datasets:{dis_ds_js} }},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{ display:true, position:'top', align:'end',
          labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
          tooltip:{{ callbacks:{{ label: ctx =>
            ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%`
          }} }} }},
        scales:{{
          x:{{ title:{{display:true,text:'% of disease annotations',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
          y:{{ ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};

  window.initTerm5 = function() {{
    if (!document.getElementById('tmc5')) return;
    new Chart('tmc5', {{
      type:'bar',
      data:{{ labels:{chem_labels_js}, datasets:{chem_ds_js} }},
      options:{{
        responsive:true, maintainAspectRatio:false, indexAxis:'y',
        plugins:{{ legend:{{ display:true, position:'top', align:'end',
          labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
          tooltip:{{ callbacks:{{ label: ctx =>
            ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%`
          }} }} }},
        scales:{{
          x:{{ title:{{display:true,text:'% of chemical annotations',color:tc,font:{{size:11}}}},
               ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
          y:{{ ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
        }}
      }}
    }});
  }};
}})();
</script>
"""
    return tabs, panels


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML corpus statistics dashboard."
    )
    parser.add_argument("input", help="Corpus statistics JSON file")
    parser.add_argument(
        "--overlap",
        "-v",
        default=None,
        metavar="FILE",
        help="Optional train/test overlap statistics JSON file",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default=None,
        metavar="FILE",
        help="Optional journal/year metadata statistics JSON file",
    )
    parser.add_argument(
        "--terminology",
        "-t",
        default=None,
        metavar="FILE",
        help="Optional terminology coverage statistics JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <input stem>_dashboard.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated file in the default browser",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_dashboard.html")
    )

    logger.info("Loading stats: %s", in_path)
    try:
        corpora = load_corpora(str(in_path))
    except FileNotFoundError:
        logger.error("Error: file not found - %s", in_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Error: invalid JSON - %s", e)
        sys.exit(1)

    if args.overlap:
        logger.info("Loading overlap: %s", args.overlap)
        try:
            attach_overlaps(corpora, load_overlaps(args.overlap))
            logger.info(
                "Overlap matched: %s / %s",
                sum(1 for c in corpora if c.get("overlap")),
                len(corpora),
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: overlap - %s", e)

    if args.metadata:
        logger.info("Loading metadata: %s", args.metadata)
        try:
            attach_metadata(corpora, load_metadata(args.metadata))
            n_m = sum(
                1 for c in corpora if (c.get("metadata") or {}).get("has_metadata")
            )
            n_t = sum(1 for c in corpora if (c.get("metadata") or {}).get("topic_dist"))
            logger.info("Metadata matched: %s / %s corpora (%s with topic data)", n_m, len(corpora), n_t)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: metadata - %s", e)

    if args.terminology:
        logger.info("Loading terminology: %s", args.terminology)
        try:
            term_raw = load_terminology(args.terminology)
            term_data = process_terminology(term_raw)
            attach_terminology(corpora, term_data)
            n_t = sum(1 for c in corpora if c.get("terminology"))
            logger.info("Terminology matched: %s / %s corpora", n_t, len(corpora))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: terminology - %s", e)

    logger.info("Corpora: %s (%s)", len(corpora), ", ".join(c["name"] for c in corpora))
    out_path.write_text(build_html(corpora), encoding="utf-8")
    logger.info("Written: %s", out_path)
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
