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
  .scope{{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:0 0 1.25rem}}
  .scope-label{{font-size:12px;color:#666;margin-right:2px}}
  .scope-btn{{border:1px solid #ddd;background:#fff;color:#555;border-radius:7px;padding:6px 10px;
        font:500 12px system-ui,-apple-system,sans-serif;cursor:pointer}}
  .scope-btn.sel{{border-color:#7F77DD;color:#111;background:#f0eefb}}
  .scope-note{{font-size:12px;color:#666;margin:-.75rem 0 1.25rem;line-height:1.5}}
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
    .scope-label,.scope-note{{color:#aaa}} .scope-btn{{background:#2a2a28;border-color:#3a3a38;color:#aaa}}
    .scope-btn.sel{{background:#363348;border-color:#7F77DD;color:#e8e6e0}}
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
  <div class="mc"><p class="ml">Corpora analyzed</p><p class="mv" id="kpiCorpora">{n_corpora}</p></div>
  <div class="mc"><p class="ml">With concept identifiers</p><p class="mv" id="kpiIds">{n_with_ids} / {n_corpora}</p></div>
  <div class="mc"><p class="ml">Ann/doc range</p><p class="mv" id="kpiAnn">{ann_min} – {ann_max}</p></div>
  <div class="mc"><p class="ml">Ambiguity range</p><p class="mv">{amb_min} – {amb_max}</p></div>
</div>

<div class="leg" id="corpusLegend">{legend_html}</div>
<div class="scope" id="scopeControls">
  <span class="scope-label">Entity scope</span>{scope_controls}
</div>
<p class="scope-note" id="scopeNote">{scope_note}</p>

<div class="tabs" id="tabs">
  <button class="tab sel" data-p="p1">Annotation density</button>
  <button class="tab" data-p="p2">Identifier coverage</button>
  <button class="tab" data-p="p3">Difficulty indicators</button>
  <button class="tab" data-p="p4">Entity type profile</button>
  <button class="tab" data-p="p5">Summary table</button>
  {overlap_tabs}{meta_tabs}{term_tabs}
</div>

<div class="panel sel" id="p1">
  <p class="sec">Annotations per thousand tokens</p>
  <div class="cw" style="height:{h_ann}px">
    <canvas id="c1" role="img" aria-label="Mean annotations per thousand tokens, log scale.">
      Annotation density per thousand tokens varies widely across corpora.
    </canvas>
  </div>
  <p class="sec" style="margin-top:1.5rem">Annotations per document</p>
  <div class="cw" style="height:{h_ann}px">
    <canvas id="c1b" role="img" aria-label="Mean annotations per document, log scale.">
      Annotation density per document varies widely across corpora.
    </canvas>
  </div>
  <p class="note">Log scale. NLM-Chem annotates full-text articles; BioID uses figure captions.</p>
</div>

<div class="panel" id="p2">
  <div id="idStatusHtml" style="margin-bottom:12px;font-size:13px">{id_status_html}</div>
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
  <tbody id="summaryRows">{table_rows}</tbody>
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
const ENTITY_PROFILES = {entity_profiles};
let currentScope = 'all';
const chartRefs = {{}};
const SCOPE_CAVEAT = 'Annotation, identifier, difficulty, and overlap views update by entity scope where scoped metric outputs are available. Terminology and metadata panels remain corpus-level.';

function prof() {{ return ENTITY_PROFILES[currentScope] || ENTITY_PROFILES.all; }}
function setMetric(el, value) {{ const node=document.getElementById(el); if(node) node.textContent=value; }}
function setHtml(el, value) {{ const node=document.getElementById(el); if(node) node.innerHTML=value; }}
function applyChart(chart, payload) {{
  if(!chart || !payload) return;
  chart.data.labels = payload.labels;
  chart.data.datasets[0].data = payload.data;
  chart.data.datasets[0].backgroundColor = payload.bg;
  chart.update();
}}
function applyCascade(datasets) {{
  const leg=document.getElementById('cascLeg');
  if(leg) leg.innerHTML=(datasets||[]).map(d=>
    `<span class="li"><span class="lc" style="background:${{d.borderColor}}"></span>${{d.label}}</span>`
  ).join('');
  if(chartRefs.c7) {{
    chartRefs.c7.data.datasets = datasets || [];
    chartRefs.c7.update();
  }}
}}
function applyScope(key) {{
  currentScope = key;
  window.currentEntityScope = key;
  const p = prof();
  document.querySelectorAll('.scope-btn').forEach(b=>b.classList.toggle('sel', b.dataset.scope===key));
  setMetric('kpiCorpora', p.nCorpora);
  setMetric('kpiIds', `${{p.nWithIds}} / ${{p.nCorpora}}`);
  setMetric('kpiAnn', `${{p.annMin}} – ${{p.annMax}}`);
  setHtml('corpusLegend', p.legendHtml);
  setHtml('idStatusHtml', p.idStatusHtml);
  setHtml('summaryRows', p.tableRows);
  setHtml('overlapRows', p.overlapRows || '');
  setHtml('scopeNote', p.description ? `${{p.description}} ${{SCOPE_CAVEAT}}` : SCOPE_CAVEAT);
  applyChart(chartRefs.c1, p.ann1k);
  applyChart(chartRefs.c1b, p.ann);
  applyChart(chartRefs.c2, p.ids);
  applyChart(chartRefs.c3, p.amb);
  applyChart(chartRefs.c4, p.variation);
  applyChart(chartRefs.c5, p.types);
  applyChart(chartRefs.c6, p.entropy);
  applyCascade(p.cascadeDatasets);
  if (window.applyTermScope) window.applyTermScope(key);
}}

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

chartRefs.c1 = new Chart('c1', {{
  type:'bar',
  data:{{ labels:{c1k_labels}, datasets:[{{ data:{c1k_data}, backgroundColor:{c1k_bg},
    borderWidth:0, borderRadius:3 }}] }},
  options:{{
    responsive:true, maintainAspectRatio:false, indexAxis:'y',
    plugins:{{ legend:{{display:false}},
      tooltip:{{ callbacks:{{ label: ctx=>` ${{ctx.parsed.x.toFixed(1)}}` }} }} }},
    scales:{{
      x:{{ type:'logarithmic',
           title:{{display:true,text:'Mean annotations per thousand tokens (log scale)',color:tc,font:{{size:11}}}},
           ticks:{{color:tc,font:{{size:11}},callback:v=>[0.1,1,10,100,1000].includes(v)?v:''}},
           grid:{{color:gc}} }},
      y:{{ ticks:{{color:tc,font:{{size:12}}}}, grid:{{color:gc}} }}
    }}
  }}
}});

chartRefs.c1b = new Chart('c1b', {{
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
function initC2(){{ chartRefs.c2 = hbar('c2',prof().ids.labels,prof().ids.data,prof().ids.bg,'Mean unique identifiers per document'); }}
function initC3(){{
  chartRefs.c3 = new Chart('c3', {{
    type:'bar',
    data:{{ labels:prof().amb.labels, datasets:[{{ data:prof().amb.data, backgroundColor:prof().amb.bg,
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
function initC4(){{ chartRefs.c4 = hbar('c4',prof().variation.labels,prof().variation.data,prof().variation.bg,'Mean surface forms per concept'); }}
function initC5(){{ chartRefs.c5 = hbar('c5',prof().types.labels,prof().types.data,prof().types.bg,'Distinct entity type labels',
  {{ticks:{{stepSize:1,color:tc,font:{{size:11}}}}}}); }}
function initC6(){{ chartRefs.c6 = hbar('c6',prof().entropy.labels,prof().entropy.data,prof().entropy.bg,'Shannon entropy (bits)'); }}

const cascadeDatasets = {cascade_datasets};
function initC7(){{
  const datasets = prof().cascadeDatasets || [];
  if (!datasets.length) return;
  const leg=document.getElementById('cascLeg');
  if(leg) leg.innerHTML=datasets.map(d=>
    `<span class="li"><span class="lc" style="background:${{d.borderColor}}"></span>${{d.label}}</span>`
  ).join('');
  chartRefs.c7 = new Chart('c7', {{
    type:'line',
    data:{{ labels:['Token vocab','Mention tokens','Mention strings','Identifiers'],
             datasets:datasets }},
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
document.getElementById('scopeControls').addEventListener('click', e=>{{
  const btn=e.target.closest('.scope-btn');
  if(!btn) return;
  applyScope(btn.dataset.scope);
}});
applyScope(currentScope);
</script>
</body>
</html>
"""
