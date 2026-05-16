import json
import yaml
from html import escape
from pathlib import Path
from .base import PALETTE, OV_COLS, BAR_SCALE, JOURNAL_TOPIC_ORDER, JOURNAL_TOPIC_COLORS, get_metric, get_stat, norm_corpus_name
from .stats import compute_entropy_from_counts, get_id_info, get_total_ann
from .terminology import _metric_scope_payload
from .templates import HTML

def load_dashboard_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return config

def _entity_scopes(config: dict | None) -> list[dict]:
    raw_scopes = (config or {}).get("entity_scopes", {})
    if not isinstance(raw_scopes, dict):
        raw_scopes = {}
    if "all" not in raw_scopes:
        raw_scopes = {"all": {"label": "All annotations", "include_all": True}, **raw_scopes}

    scopes = []
    for key, raw_scope in raw_scopes.items():
        if not isinstance(raw_scope, dict):
            continue
        scopes.append(
            {
                "key": str(key),
                "label": str(raw_scope.get("label") or key).strip(),
                "description": str(raw_scope.get("description") or "").strip(),
                "include_all": bool(raw_scope.get("include_all", False)),
                "labels": {str(label) for label in raw_scope.get("labels", [])},
            }
        )
    return scopes

def _scope_label_counts(corpus, scope):
    scope_key = None if scope.get("include_all") else scope.get("key")
    if scope_key:
        details = get_metric(
            corpus.get("metric_results") or [],
            "label_distribution",
            "details",
            scope=scope_key,
        ) or {}
        counts = details.get("counts", {})
        if counts:
            return counts
    counts = dict(corpus.get("label_counts") or {})
    if scope.get("include_all"):
        return counts
    labels = scope.get("labels") or set()
    return {label: count for label, count in counts.items() if label in labels}

def _scope_corpus(corpus, scope):
    scope_key = None if scope.get("include_all") else scope.get("key")
    metrics = corpus.get("metric_results") or []
    scoped_counts = _scope_label_counts(corpus, scope)
    total_ann = sum(scoped_counts.values())
    scoped = dict(corpus)
    scoped["n_types"] = len(scoped_counts)
    scoped["types"] = list(scoped_counts)
    scoped["label_counts"] = scoped_counts
    scoped["total_ann"] = total_ann
    scoped["ann_per_doc"] = round(
        get_stat(
            metrics,
            "annotations_per_document_stats",
            "mean",
            total_ann / corpus["doc_count"] if corpus["doc_count"] else 0,
            scope=scope_key,
        ),
        2,
    )
    scoped["ann_per_1k"] = round(
        get_stat(metrics, "annotations_per_1000_tokens_stats", "mean", 0, scope=scope_key),
        2,
    )
    scoped["men_per_doc"] = round(
        get_stat(
            metrics,
            "unique_mentions_per_document_stats",
            "mean",
            scoped.get("men_per_doc", 0),
            scope=scope_key,
        ),
        2,
    )
    scoped["ids_per_doc"] = round(
        get_stat(
            metrics,
            "unique_identifiers_per_document_stats",
            "mean",
            scoped.get("ids_per_doc", 0) or 0,
            scope=scope_key,
        ),
        2,
    )
    info = get_id_info(metrics, scope=scope_key)
    if metrics and any(m.get("metric_name") == "identifier_resource_distribution" for m in metrics):
        scoped["id_vocab"] = info["label"]
        scoped["id_class"] = info["css_class"]
        scoped["has_ids"] = info["has_ids"]
    scoped["ambiguity"] = round(
        get_stat(metrics, "ambiguity_degree_stats", "mean", scoped.get("ambiguity"), scope=scope_key),
        3,
    )
    scoped["variation"] = get_stat(
        metrics,
        "variation_degree_stats",
        "mean",
        scoped.get("variation"),
        scope=scope_key,
    )
    if not scoped["has_ids"]:
        scoped["variation"] = None
    if scope_key and corpus.get("overlap"):
        scoped["overlap"] = (corpus["overlap"].get("scopes") or {}).get(scope_key)
    else:
        scoped["overlap"] = corpus.get("overlap")
    scoped["entropy"] = round(compute_entropy_from_counts(scoped_counts), 2)
    return scoped

def _scoped_corpora(corpora, scope):
    scoped = [_scope_corpus(corpus, scope) for corpus in corpora]
    if scope.get("include_all"):
        return scoped
    return [corpus for corpus in scoped if corpus["total_ann"] > 0]

def _hbar_payload(corpora, key, colours, *, sorted_values=True, include_null=False):
    pairs = [
        (c["name"], c.get(key), colours[c["color_index"] % len(colours)])
        for c in corpora
        if include_null or c.get(key) is not None
    ]
    if sorted_values:
        pairs.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))
    return {
        "labels": [p[0] for p in pairs],
        "data": [p[1] if p[1] is not None else 0 for p in pairs],
        "bg": [
            col if (val is not None and val > 0) else col + "33"
            for _, val, col in pairs
        ],
    }

def _scope_controls(scopes):
    return "".join(
        f'<button class="scope-btn{" sel" if i == 0 else ""}" data-scope="{scope["key"]}">{scope["label"]}</button>'
        for i, scope in enumerate(scopes)
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

def cascade_datasets(corpora, colours):
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
        col = colours[c.get("color_index", i) % len(colours)]
        ds.append(
            {
                "label": c["name"],
                "data": pct,
                "borderColor": col,
                "backgroundColor": col,
                "pointRadius": [4 if v is not None else 0 for v in pts],
                "pointHoverRadius": [6, 6, 6, 6],
                "borderWidth": 2,
                "spanGaps": False,
                "tension": 0.1,
            }
        )
    return ds

def cascade_datasets_js(corpora, colours):
    return json.dumps(cascade_datasets(corpora, colours))

def build_overlap_rows(corpora):
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
    return "".join(rows)

def build_overlap_panels(corpora):
    oc = OV_COLS
    tabs = (
        '\n  <button class="tab" data-p="p6">Train-test overlap</button>'
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
        f'<tbody id="overlapRows">{build_overlap_rows(corpora)}</tbody></table></div>'
        f'<div class="fn">All values are Jaccard similarity (intersection / union) between splits.</div>'
        f'<p class="sec" style="margin-top:1.5rem">Overlap cascade</p>'
        f'<div class="leg" id="cascLeg"></div>'
        f'<div class="cw" style="height:380px">'
        f'<canvas id="c7" role="img" aria-label="Overlap cascade across four abstraction levels.">'
        f"Overlap cascade from token vocabulary to identifier level.</canvas></div>"
        f'<p class="note">Each line traces one corpus across four abstraction levels. '
        f"Lines that terminate before the identifier level indicate corpora without concept normalization.</p>"
        f"</div>"
    )
    return tabs, panels

def build_legend_html(corpora, colours, *, use_color_index=False):
    return "".join(
        f'<span class="li"><span class="lc" style="background:{colours[(c.get("color_index", i) if use_color_index else i) % len(colours)]}"></span>'
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

def _entity_profile_data(corpora, colours, config):
    scopes = _entity_scopes(config)
    data = {}
    for scope in scopes:
        scoped = _scoped_corpora(corpora, scope)
        for corpus in scoped:
            corpus["color_index"] = next(
                (i for i, original in enumerate(corpora) if original["raw_name"] == corpus["raw_name"]),
                0,
            )

        data[scope["key"]] = {
            "label": scope["label"],
            "description": scope["description"],
            "nCorpora": len(scoped),
            "nWithIds": sum(1 for c in scoped if c["has_ids"]),
            "annMin": f"{min((c['ann_per_doc'] for c in scoped), default=0):.1f}",
            "annMax": f"{max((c['ann_per_doc'] for c in scoped), default=0):.1f}",
            "legendHtml": build_legend_html(scoped, colours, use_color_index=True),
            "idStatusHtml": build_id_status_html(scoped),
            "tableRows": build_table_rows(scoped),
            "overlapRows": build_overlap_rows(scoped),
            "cascadeDatasets": cascade_datasets(scoped, colours),
            "ann1k": _hbar_payload(scoped, "ann_per_1k", colours),
            "ann": _hbar_payload(scoped, "ann_per_doc", colours),
            "ids": _hbar_payload(scoped, "ids_per_doc", colours),
            "amb": _hbar_payload(
                [c for c in scoped if c["has_ids"]],
                "ambiguity",
                colours,
                sorted_values=False,
            ),
            "variation": _hbar_payload(
                [c for c in scoped if c["has_ids"]],
                "variation",
                colours,
            ),
            "types": _hbar_payload(scoped, "n_types", colours),
            "entropy": _hbar_payload(scoped, "entropy", colours),
        }
    return data

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(ch * 2 for ch in hex_color)
    if len(hex_color) != 6:
        return (127, 119, 221)
    try:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (127, 119, 221)


def build_topic_heatmap(corpora, metadata_key: str = "topic_dist") -> str:
    """Generate an HTML heatmap: rows = topics, columns = corpora with topic data."""
    with_td = sorted(
        [c for c in corpora if (c.get("metadata") or {}).get(metadata_key)],
        key=lambda c: c["name"],
    )
    if not with_td:
        return "<p style='color:var(--color-text-secondary);font-size:13px'>No topic data available.</p>"

    observed_topics = {
        topic for c in with_td for topic in c["metadata"][metadata_key].keys()
    }
    ordered_topics = [
        topic for topic in JOURNAL_TOPIC_ORDER if topic in observed_topics
    ] + sorted(observed_topics - set(JOURNAL_TOPIC_ORDER))
    shown_topics = [
        topic
        for topic in ordered_topics
        if max(c["metadata"][metadata_key].get(topic, 0.0) for c in with_td) >= 1.0
    ]
    if not shown_topics:
        return "<p style='color:var(--color-text-secondary);font-size:13px'>No topic data available.</p>"

    heatmap_color = "#7F77DD"
    heatmap_rgb = _hex_to_rgb(heatmap_color)
    heatmap_max = max(
        c["metadata"][metadata_key].get(topic, 0.0)
        for c in with_td
        for topic in shown_topics
    )
    grid_template = (
        "grid-template-columns:minmax(210px,1.35fr) "
        f"repeat({len(with_td)},minmax(88px,1fr))"
    )
    header = (
        '<div class="hm-corner">Topic</div>'
        + "".join(
            f'<div class="hm-col" title="{escape(c["name"])}">{escape(c["name"])}</div>'
            for c in with_td
        )
    )
    rows = []
    for topic in shown_topics:
        vals = [c["metadata"][metadata_key].get(topic, 0.0) for c in with_td]
        col = JOURNAL_TOPIC_COLORS.get(topic, "#D3D1C7")
        mx = max(vals)
        dot = (
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
            f'background:{col};margin-right:6px;vertical-align:middle"></span>'
        )
        cells = []
        for v in vals:
            if v < 1:
                cells.append('<div class="hm-cell hm-zero" title="Less than 1%">—</div>')
                continue
            intensity = min(v / heatmap_max, 1.0) if heatmap_max else 0.0
            red = round(255 + (heatmap_rgb[0] - 255) * intensity)
            green = round(255 + (heatmap_rgb[1] - 255) * intensity)
            blue = round(255 + (heatmap_rgb[2] - 255) * intensity)
            text_color = "#fff" if intensity >= 0.58 else "#111"
            cells.append(
                '<div class="hm-cell" '
                f'style="background:rgb({red},{green},{blue});'
                f'color:{text_color};'
                f'font-weight:{"650" if v == mx else "500"}" '
                f'title="{escape(topic)}: {v:.1f}%">'
                f'{v:.0f}%</div>'
            )
        rows.append(
            f'<div class="hm-row" style="{grid_template}">'
            f'<div class="hm-topic">{dot}{escape(topic)}</div>'
            f'{"".join(cells)}</div>'
        )

    total_cells = []
    for c in with_td:
        shown = sum(
            c["metadata"][metadata_key].get(t, 0)
            for t in shown_topics
        )
        total_cells.append(f'<div class="hm-total-cell">{shown:.0f}%</div>')

    return f"""
<div class="topic-heatmap-wrap">
  <div class="topic-heatmap">
    <div class="hm-head" style="{grid_template}">
      {header}
    </div>
    {"".join(rows)}
    <div class="hm-total" style="{grid_template}">
      <div>Total shown</div>{"".join(total_cells)}
    </div>
  </div>
  <div class="hm-scale" aria-hidden="true">
    <span>Lower</span><span class="hm-ramp"></span><span>Higher within topic</span>
  </div>
</div>"""


def build_topic_table(corpora, metadata_key: str = "topic_dist") -> str:
    return build_topic_heatmap(corpora, metadata_key)

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
    article_topic_table = build_topic_table(corpora, "article_topic_dist")

    tabs = (
        '\n  <button class="tab" data-p="p8">Journal metadata</button>'
        '\n  <button class="tab" data-p="p9">Temporal coverage</button>'
        '\n  <button class="tab" data-p="p11">Article topics</button>'
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

<div class="panel" id="p11">
  <p class="sec">Article topic distribution per corpus (%)</p>
  {article_topic_table}
  <div class="fn">
    Topics are high-level MeSH-derived article categories resolved from article
    metadata MeSH terms, with unresolved article-term fractions filled from journal
    MeSH topics and configured journal-name fallback topics. Only topics with ≥ 1%
    share in at least one corpus are shown. Dominant value per row is bold.
    Percentages may not sum to exactly
    100 due to rounding.
  </div>
</div>

<div class="panel" id="p10">
  <p class="sec">Journal topic distribution per corpus (%)</p>
  {topic_table}
  <div class="fn">
    Topics are high-level MeSH-derived journal categories resolved from the journal
    record's NLM Catalog MeSH topics, with configured journal-name fallback topics for
    journals that do not have MeSH topics. Only topics with ≥ 1% share in at least one
    corpus are shown. Dominant value per row is bold. Percentages may not sum to exactly
    100 due to rounding.
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

def _terminology_profiles(term_data):
    scopes = {"all"}
    for corpus_data in term_data.values():
        scopes.update((corpus_data.get("by_scope") or {}).keys())
    profiles = {}
    entries_by_scope = {
        scope: [
            entry
            for corpus_data in term_data.values()
            for entry in (corpus_data.get("by_scope") or {}).get(scope, [])
            if entry.get("n_input_ids", 0) > 0
        ]
        for scope in scopes
    }
    for scope in sorted(scopes):
        entries = entries_by_scope.get(scope, [])
        chart_entries = entries
        if scope == "all":
            scoped_entries = [
                entry
                for scope_key, scope_entries in entries_by_scope.items()
                if scope_key != "all"
                for entry in scope_entries
            ]
            if scoped_entries:
                chart_entries = scoped_entries
        labels = [entry["series_label"] for entry in entries]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(entries))]
        table_rows = "".join(
            "<tr>"
            f"<td class='l'><strong>{entry['display_name']}</strong></td>"
            f"<td class='l'>{entry['terminology_label']}</td>"
            f"<td class='r'>{entry['n_input_ids']:,}</td>"
            f"<td class='r'>{entry['n_missing_ids']:,} ({entry['missing_pct']:.2f}%)</td>"
            f"<td class='r'><strong>{entry['coverage_pct']:.2f}%</strong></td>"
            "</tr>"
            for entry in entries
        )
        if not table_rows:
            table_rows = "<tr><td class='l' colspan='5'>No terminology data for this entity scope.</td></tr>"

        entries_by_group = {}
        for entry in chart_entries:
            group_key = (entry.get("entity_scope") or scope, entry["terminology"])
            entries_by_group.setdefault(group_key, []).append(entry)

        chart_groups = []
        for (scope_key, terminology_name), terminology_entries in sorted(
            entries_by_group.items(),
            key=lambda item: (item[1][0].get("entity_scope_label") or item[0][0], item[1][0]["terminology_label"]),
        ):
            group_colors = [PALETTE[i % len(PALETTE)] for i in range(len(terminology_entries))]
            depth_labels = sorted(
                {
                    int(depth)
                    for entry in terminology_entries
                    for depth in entry.get("depth", {})
                    if str(depth).isdigit()
                }
            )
            depth_datasets = []
            for i, entry in enumerate(terminology_entries):
                depth_datasets.append(
                    {
                        "label": entry["display_name"],
                        "data": [
                            round(entry["depth"].get(str(depth), {}).get("proportion", 0) * 100, 2)
                            for depth in depth_labels
                        ],
                        "borderColor": group_colors[i],
                        "backgroundColor": group_colors[i] + "22",
                        "fill": False,
                        "borderWidth": 2,
                        "pointRadius": 4,
                        "tension": 0.3,
                    }
                )
            depth_note = "Mean annotation depth: " + " | ".join(
                f"{entry['display_name']} {entry['mean_depth']}" for entry in terminology_entries
            ) if terminology_entries else "No terminology data for this entity scope."

            branch_codes = sorted(
                {
                    code
                    for entry in terminology_entries
                    for code, branch in entry.get("branches", {}).items()
                    if branch.get("configured_anchor") or branch.get("total", 0) > 0 or branch.get("proportion", 0) > 0
                },
                key=lambda code: -sum(entry.get("branches", {}).get(code, {}).get("proportion", 0) for entry in terminology_entries),
            )
            branch_labels = []
            for code in branch_codes:
                label = next(
                    (
                        entry["branches"][code].get("label")
                        for entry in terminology_entries
                        if code in entry.get("branches", {})
                    ),
                    code,
                )
                branch_labels.append(label if label == code else f"{code} {label}")
            recall_datasets = []
            annotation_datasets = []
            for i, entry in enumerate(terminology_entries):
                recall_datasets.append(
                    {
                        "label": entry["display_name"],
                        "data": [
                            round(entry.get("branches", {}).get(code, {}).get("proportion", 0) * 100, 2)
                            for code in branch_codes
                        ],
                        "backgroundColor": group_colors[i] + "bb",
                        "borderWidth": 0,
                        "borderRadius": 2,
                    }
                )
                annotation_datasets.append(
                    {
                        "label": entry["display_name"],
                        "data": [
                            round(entry.get("branches", {}).get(code, {}).get("annotation_proportion", 0) * 100, 2)
                            for code in branch_codes
                        ],
                        "backgroundColor": group_colors[i] + "bb",
                        "borderWidth": 0,
                        "borderRadius": 2,
                    }
                )
            chart_groups.append(
                {
                    "key": f"{scope_key}:{terminology_name}",
                    "title": f"{terminology_entries[0].get('entity_scope_label') or scope_key} / {terminology_entries[0]['terminology_label']}",
                    "depth": {
                        "labels": depth_labels,
                        "datasets": depth_datasets,
                        "note": depth_note,
                    },
                    "recall": {
                        "labels": branch_labels,
                        "datasets": recall_datasets,
                        "height": max(320, len(branch_labels) * 44 + 120),
                    },
                    "annotation": {
                        "labels": branch_labels,
                        "datasets": annotation_datasets,
                        "height": max(320, len(branch_labels) * 44 + 120),
                    },
                }
            )

        profiles[scope] = {
            "tableRows": table_rows,
            "coverage": {
                "labels": labels,
                "found": [entry["coverage_pct"] for entry in entries],
                "missing": [entry["missing_pct"] for entry in entries],
                "colors": colors,
                "height": max(240, len(labels) * 34 + 110),
            },
            "chartGroups": chart_groups,
        }
    return profiles

def build_terminology_panels(term_data):
    """
    term_data: dict keyed by norm_corpus_name(corpus_name) → processed stats dict
    Returns (tabs_html, panels_html).
    """
    if not term_data:
        return "", ""

    tabs = (
        '\n  <button class="tab" data-p="pterm1">Deprecated terms</button>'
        '\n  <button class="tab" data-p="pterm3">Annotation depth</button>'
        '\n  <button class="tab" data-p="pterm4">Terminology coverage</button>'
        '\n  <button class="tab" data-p="pterm5">Annotation topic coverage</button>'
    )
    profiles = _terminology_profiles(term_data)

    panels = f"""
<div class="panel" id="pterm1">
  <p class="sec">Deprecated terms summary</p>
  <div style="overflow-x:auto;margin-bottom:1.5rem">
  <table><thead><tr>
    <th class="l">Corpus</th>
    <th class="l">Terminology</th>
    <th class="r">Total concepts</th>
    <th class="r">Deprecated concepts</th>
    <th class="r">Resolvable identifier rate</th>
  </tr></thead><tbody id="termCoverageRows"></tbody></table>
  </div>
  <p class="sec">Resolvable identifier rate</p>
  <div class="cw" id="termCoverageChartWrap" style="height:240px">
    <canvas id="tmc1" role="img" aria-label="Horizontal stacked bar: vocabulary coverage per corpus.">
      Coverage rates for all corpora.
    </canvas>
  </div>
  <div class="fn">
    Coverage counts only identifiers whose resource is associated with the selected terminology.
  </div>
</div>

<div class="panel" id="pterm3">
  <p class="sec">Annotation depth distribution</p>
  <div id="termDepthCharts"></div>
</div>

<div class="panel" id="pterm4">
  <p class="sec">Terminology coverage</p>
  <div id="termTerminologyCoverageCharts"></div>
  <p class="note">Terminology coverage = unique corpus concept count in branch ÷ total terminology concepts in that branch.
  Only branches with signal in the selected scope are shown.</p>
</div>

<div class="panel" id="pterm5">
  <p class="sec">Annotation topic coverage</p>
  <div id="termAnnotationCharts"></div>
  <p class="note">Annotation topic coverage = annotation-weighted branch count ÷ all identifiers for that corpus and entity scope, including deprecated identifiers in the denominator.</p>
</div>

<script>
(function() {{
  const dk = matchMedia('(prefers-color-scheme:dark)').matches;
  const tc = dk ? 'rgba(255,255,255,0.55)' : 'rgba(0,0,0,0.45)';
  const gc = dk ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.05)';

  const TERM_PROFILES = {json.dumps(profiles)};
  const palette = {json.dumps(PALETTE)};
  const charts = {{}};

  function currentProfile(scope) {{
    return TERM_PROFILES[scope] || TERM_PROFILES.all || {{entries:[], coverage:null, chartGroups:[], tableRows:''}};
  }}
  function noData(id, message) {{
    const el = document.getElementById(id);
    if (el) el.innerHTML = message || 'No terminology data for this entity scope.';
  }}
  function destroyCharts(prefix) {{
    Object.keys(charts).forEach(id => {{
      if (id.startsWith(prefix)) {{
        charts[id].destroy();
        delete charts[id];
      }}
    }});
  }}
  function updateBar(id, config) {{
    const canvas = document.getElementById(id);
    const labels = config && config.data && config.data.labels;
    if (!canvas || !labels || !labels.length) return false;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(canvas, config);
    return true;
  }}

  function renderTerm1(scope) {{
    const p = currentProfile(scope);
    const rows = document.getElementById('termCoverageRows');
    if (rows) rows.innerHTML = p.tableRows || '';
    if (!p.coverage || !p.coverage.labels.length) {{
      if (charts.tmc1) {{ charts.tmc1.destroy(); delete charts.tmc1; }}
      return;
    }}
    const wrap = document.getElementById('termCoverageChartWrap');
    if (wrap) wrap.style.height = `${{p.coverage.height || 240}}px`;
    updateBar('tmc1', {{
      type:'bar',
      data:{{
        labels:p.coverage.labels,
        datasets:[
          {{ label:'Current (%)', data:p.coverage.found,
             backgroundColor:p.coverage.colors.map(c=>c+'bb'), borderWidth:0, borderRadius:3 }},
          {{ label:'Deprecated (%)', data:p.coverage.missing,
             backgroundColor:p.coverage.colors.map(()=>'#E24B4A44'), borderWidth:0, borderRadius:3 }}
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
  }}

  function renderTerm3(scope) {{
    const p = currentProfile(scope);
    const root = document.getElementById('termDepthCharts');
    destroyCharts('tmc3_');
    if (!root) return;
    const groups = (p.chartGroups || []).filter(group => group.depth && group.depth.labels.length);
    if (!groups.length) {{
      root.innerHTML = '<div class="fn">No terminology data for this entity scope.</div>';
      return;
    }}
    root.innerHTML = groups.map((group, i) => `
      <p class="sec">${{group.title}}</p>
      <div class="cw" style="height:300px">
        <canvas id="tmc3_${{i}}" role="img" aria-label="Line chart: ${{group.title}} depth distribution comparison.">
          Depth distribution for ${{group.title}}.
        </canvas>
      </div>
      <div class="fn">${{group.depth.note || ''}}</div>
    `).join('');
    groups.forEach((group, i) => {{
      updateBar(`tmc3_${{i}}`, {{
        type:'line',
        data:{{ labels:group.depth.labels, datasets:group.depth.datasets }},
        options:{{
          responsive:true, maintainAspectRatio:false,
          plugins:{{ legend:{{ display:true, position:'top', align:'end',
            labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
            tooltip:{{ callbacks:{{ label: ctx =>
              ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}%`
            }} }} }},
          scales:{{
            x:{{ title:{{display:true,text:'Ontology hierarchy depth',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }},
            y:{{ min:0,
                 title:{{display:true,text:'% of annotations',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }}
          }}
        }}
      }});
    }});
  }}

  function renderTerm4(scope) {{
    const p = currentProfile(scope);
    const root = document.getElementById('termTerminologyCoverageCharts');
    destroyCharts('tmc4_');
    if (!root) return;
    const groups = (p.chartGroups || []).filter(group => group.recall && group.recall.labels.length);
    if (!groups.length) {{
      root.innerHTML = '<div class="fn">No terminology data for this entity scope.</div>';
      return;
    }}
    root.innerHTML = groups.map((group, i) => `
      <p class="sec">${{group.title}}</p>
      <div class="cw" style="height:${{group.recall.height}}px">
        <canvas id="tmc4_${{i}}" role="img" aria-label="Grouped horizontal bar: ${{group.title}} branch recall.">
          Terminology branch recall for ${{group.title}}.
        </canvas>
      </div>
    `).join('');
    groups.forEach((group, i) => {{
      updateBar(`tmc4_${{i}}`, {{
        type:'bar',
        data:{{ labels:group.recall.labels, datasets:group.recall.datasets }},
        options:{{
          responsive:true, maintainAspectRatio:false, indexAxis:'y',
          plugins:{{ legend:{{ display:true, position:'top', align:'end',
            labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
            tooltip:{{ callbacks:{{ label: ctx =>
              ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%`
            }} }} }},
          scales:{{
            x:{{ title:{{display:true,text:'% of terminology branch concepts covered',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
            y:{{ ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
          }}
        }}
      }});
    }});
  }}

  function renderTerm5(scope) {{
    const p = currentProfile(scope);
    const root = document.getElementById('termAnnotationCharts');
    destroyCharts('tmc5_');
    if (!root) return;
    const groups = (p.chartGroups || []).filter(group => group.annotation && group.annotation.labels.length);
    if (!groups.length) {{
      root.innerHTML = '<div class="fn">No terminology data for this entity scope.</div>';
      return;
    }}
    root.innerHTML = groups.map((group, i) => `
      <p class="sec">${{group.title}}</p>
      <div class="cw" style="height:${{group.annotation.height}}px">
        <canvas id="tmc5_${{i}}" role="img" aria-label="Grouped horizontal bar: ${{group.title}} annotation topic coverage.">
          Annotation topic coverage for ${{group.title}}.
        </canvas>
      </div>
    `).join('');
    groups.forEach((group, i) => {{
      updateBar(`tmc5_${{i}}`, {{
        type:'bar',
        data:{{ labels:group.annotation.labels, datasets:group.annotation.datasets }},
        options:{{
          responsive:true, maintainAspectRatio:false, indexAxis:'y',
          plugins:{{ legend:{{ display:true, position:'top', align:'end',
            labels:{{ boxWidth:10,boxHeight:10,borderRadius:2,font:{{size:11}},color:tc }} }},
            tooltip:{{ callbacks:{{ label: ctx =>
              ` ${{ctx.dataset.label}}: ${{ctx.parsed.x.toFixed(1)}}%`
            }} }} }},
          scales:{{
            x:{{ min:0, max:100,
                 title:{{display:true,text:'% of corpus identifiers in topic',color:tc,font:{{size:11}}}},
                 ticks:{{color:tc,font:{{size:11}},callback:v=>v+'%'}}, grid:{{color:gc}} }},
            y:{{ ticks:{{color:tc,font:{{size:11}}}}, grid:{{color:gc}} }}
          }}
        }}
      }});
    }});
  }}

  window.applyTermScope = function(scope) {{
    renderTerm1(scope || 'all');
    renderTerm3(scope || 'all');
    renderTerm4(scope || 'all');
    renderTerm5(scope || 'all');
  }};
  window.initTerm1 = () => renderTerm1(window.currentEntityScope || 'all');
  window.initTerm3 = () => renderTerm3(window.currentEntityScope || 'all');
  window.initTerm4 = () => renderTerm4(window.currentEntityScope || 'all');
  window.initTerm5 = () => renderTerm5(window.currentEntityScope || 'all');
}})();
</script>
"""
    return tabs, panels

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

def build_html(corpora, dashboard_config=None):
    colours = PALETTE[:]
    for i, corpus in enumerate(corpora):
        corpus["color_index"] = i
    entity_profiles = _entity_profile_data(corpora, colours, dashboard_config or {})
    scopes = _entity_scopes(dashboard_config or {})
    default_profile = entity_profiles.get("all") or next(iter(entity_profiles.values()))
    n = len(corpora)
    has_ov = any(c.get("overlap") for c in corpora)
    has_meta = any(c.get("metadata") for c in corpora)

    ann_vals = [c["ann_per_doc"] for c in corpora]
    amb_vals = [c["ambiguity"] for c in corpora]
    h_ann = max(300, n * 40 + 80)
    amb_lo = round(max(0.95, min(amb_vals) - 0.02), 2)
    amb_hi = round(max(amb_vals) + 0.02, 2)

    c1k_l, c1k_d, c1k_b = _sorted_hbar(corpora, "ann_per_1k", colours)
    c1_l, c1_d, c1_b = _sorted_hbar(corpora, "ann_per_doc", colours)
    c2_l, c2_d, c2_b = _sorted_hbar(corpora, "ids_per_doc", colours)
    c3_l, c3_d, c3_b = _all_hbar(corpora, "ambiguity", colours)
    c4_l, c4_d, c4_b = _variation_data(corpora, colours)
    c5_l, c5_d, c5_b = _sorted_hbar(corpora, "n_types", colours)
    c6_l, c6_d, c6_b = _sorted_hbar(corpora, "entropy", colours)

    if has_ov:
        ov_tabs, ov_panels = build_overlap_panels(corpora)
        cascade_ds = json.dumps(cascade_datasets(corpora, colours))
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
            norm_corpus_name(c["raw_name"]): c["terminology"]
            for c in corpora
            if c.get("terminology")
        }
        term_tabs, term_panels = build_terminology_panels(term_data_for_panels)
        term_panel_js = (
            "pterm1:window.initTerm1,\n  pterm3:window.initTerm3,"
            "\n  pterm4:window.initTerm4,\n  pterm5:window.initTerm5,"
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
        scope_controls=_scope_controls(scopes),
        scope_note=default_profile.get("description") or "Annotation, identifier, difficulty, and overlap views update by entity scope where scoped metric outputs are available. Terminology and metadata panels remain corpus-level.",
        entity_profiles=json.dumps(entity_profiles),
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
        c1k_labels=c1k_l,
        c1k_data=c1k_d,
        c1k_bg=c1k_b,
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
