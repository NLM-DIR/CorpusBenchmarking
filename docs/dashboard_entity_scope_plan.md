# Dashboard Entity Scope Plan

## Problem

The dashboard currently presents each corpus as one undifferentiated annotation set. That is misleading for corpora such as BioID, CRAFT, AnatEM, and JNLPBA because they contain many entity categories beyond diseases, chemicals, and cell types. The terminology coverage panel is also hardcoded around disease and chemical views, so it cannot clearly explain mappings such as CRAFT MONDO annotations being disease-like even though the corpus label vocabulary is not identical to MeSH disease labels.

## First implementation

Add a dashboard configuration file that defines named entity scopes. Each scope maps dashboard-level entity categories to the raw labels present in `label_distribution`. The dashboard should load this configuration and expose a scope selector in the generated HTML.

For each selected scope, update the label-derived parts of the dashboard:

- corpora shown in the main overview
- scoped annotation totals
- scoped annotations per document
- number of matching labels
- scoped label entropy
- summary table rows

Metrics that are not currently available per entity type should remain corpus-level for now:

- identifier vocabulary and identifier density
- ambiguity
- variation
- train/test overlap
- terminology coverage
- journal metadata

The UI should state this explicitly so the scoped view does not imply unavailable per-entity recomputation.

## Configuration shape

Use `configs/dashboard.yaml`:

```yaml
entity_scopes:
  all:
    label: All annotations
    include_all: true
  disease:
    label: Diseases
    labels: ["Disease", "Cancer", "Pathological_formation"]
    notes:
      - "CRAFT Disease comes from MONDO annotations; this is comparable as disease-like scope, but not a MeSH label identity."
```

The config can grow later with per-corpus overrides or terminology mappings if a label needs different treatment in different corpora.

## Later work

- Add filtered benchmark outputs so ambiguity, variation, identifier density, and overlaps can be recomputed per entity scope.
- Move HTML/CSS/JS templates out of the Python module.
- Split dashboard building into data preparation, chart specification, and HTML rendering modules.
- Make terminology coverage configurable by entity scope instead of hardcoded disease/chemical panels.
