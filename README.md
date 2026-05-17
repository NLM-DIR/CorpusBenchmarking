# CorpusBenchmarking

**CorpusBenchmarking** is a corpus-centric Python framework for diagnosing the benchmark utility of biomedical named entity recognition (NER) and entity linking (EL) corpora. It treats corpora as measurement instruments rather than fixed inputs, and reports corpus-intrinsic properties that affect what benchmark results can and cannot support.

**Dashboard:** the current dashboard for the preconfigured corpora and terminologies is published at <https://nlm-dir.github.io/CorpusBenchmarking/dashboard.html>.

The framework characterizes corpora across several diagnostic families:

* **Scale and density**: document counts, token counts, annotation density, and related corpus-size summaries.
* **Lexical and conceptual structure**: mention ambiguity, surface-form variation, and concept reuse.
* **Label distribution**: entity-label balance and Shannon entropy.
* **Overlap and independence**: train-test overlap at mention, normalized identifier, and document levels.
* **Metadata composition**: publication-year coverage, journal diversity, journal concentration, and high-level article/journal topic profiles.
* **Ontology coverage**: concept coverage, ontology depth, and high-level terminology branch coverage for supported terminologies.

## Features

* **Standardized representation**: Both NER-only corpora and NER+EL corpora in diverse source formats are converted into a common model of documents, passages, annotations, labels, and identifier links. Converted corpus is cached in JSON for fast reload.
* **Corpus format support**: Built-in loaders include BioC XML, PubTator, BRAT-style standoff annotation, and CRAFT Knowtator XML. Supports raw downloads and zip, tar, tar.gz, and gzip archives.
* **Terminology format support**: Terminology loaders support MeSH XML and OBO ontology files, including downloadable OBO resources such as CL, MONDO, and ChEBI.
* **Automatic acquisition**: Corpus configs can declare source URLs, archive formats, and converters. The runner downloads and extracts missing corpora when needed, then caches parsed corpora for later runs.
* **Local-file friendly**: If loader paths already point to local files, no download step is required. This supports manually prepared corpora, private datasets, and reproducible local snapshots.
* **Registry-based architecture**: Corpus loaders, terminology loaders, metadata fetchers, converters, and metrics are registered under stable names and selected from YAML.
* **Metadata enrichment**: PubMed/PMC metadata fetchers populate a persistent JSON metadata store for article and journal metadata.
* **Configurable filtering and scopes**: Annotation filters and dashboard entity scopes allow analyses over all annotations or focused entity groups such as diseases, chemicals, cells, anatomy, and genes/proteins.
* **Interactive dashboard**: The pipeline writes structured JSON outputs and a self-contained HTML dashboard with metadata panels, topic heatmaps, overlap summaries, terminology coverage panels, and configurable entity-scope controls.

## Preconfigured Corpora and Terminologies

The default metric batteries currently preconfigure nine biomedical NER or NER+EL corpora. Entity scopes are dashboard filters defined in `configs/dashboard.yaml`; terminology coverage is computed only where corpus identifiers can be resolved against a configured terminology.

| Corpus | Entity scopes available in the dashboard | Terminology coverage configured |
| --- | --- | --- |
| AnatEM | Anatomy; cells and cell states; diseases | None; this is an NER-only standoff corpus in the current configuration. |
| BC5CDR | Chemicals; diseases | MeSH 2026 for chemical and disease identifiers. |
| BioID | Anatomy; cells and cell states; chemicals; functions and processes; genes/proteins/sequences; species | Cell Ontology (CL) for cell annotations; ChEBI for chemical annotations. |
| CHEMDNER | Chemicals | None; this configuration filters chemical mentions but does not provide normalized terminology identifiers for coverage metrics. |
| CRAFT | Anatomy; cells and cell states; chemicals; diseases; functions and processes; genes/proteins/sequences; species | Cell Ontology (CL), MONDO, and ChEBI for the corresponding cell, disease, and chemical scopes. The corpus also carries identifiers from GO, NCBITaxon, PR, SO, UBERON, and related CRAFT annotation resources. |
| CellLink | Cells and cell states | Cell Ontology (CL) for cell-state and cell-line/cell-type links. |
| JNLPBA | Cells and cell states; genes/proteins/sequences | None; this is an NER-only standoff corpus in the current configuration. |
| NCBI Disease | Diseases | MeSH 2026 for disease identifiers. OMIM identifiers are retained in resource-distribution metrics but are not loaded as a terminology coverage resource by default. |
| NLM-Chem | Chemicals | MeSH 2026 for chemical identifiers. |

The terminology coverage battery loads four terminology resources: MeSH XML for MeSH 2026, OBO Cell Ontology (`CL`), OBO MONDO, and OBO ChEBI. High-level terminology branch mappings are configured under `configs/MeSH_*_mappings.yaml`, `configs/cell_ontology_mappings.yaml`, `configs/MONDO_disease_mappings.yaml`, and `configs/ChEBI_chemical_mappings.yaml`.

## Install

This framework requires Python 3.11 or higher.

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the full diagnostic pipeline and regenerate the dashboard:

```bash
bash scripts/update_output.sh
```

The script runs the configured metric batteries for basic corpus statistics, overlap, metadata, and terminology coverage, then writes `docs/dashboard.html`.

Individual batteries can also be run directly:

```bash
PYTHONPATH=src python -u src/corpus_benchmark/cli.py configs/basic_corpus_stats.yaml
PYTHONPATH=src python -u src/corpus_benchmark/cli.py configs/metadata_stats.yaml
PYTHONPATH=src python -u src/corpus_benchmark/cli.py configs/terminology_coverage.yaml
```

No separate data-preparation step is required when using the provided configs: missing downloadable resources are acquired automatically. For local-only use, point the corpus or terminology loader paths at files already present on disk and omit the acquisition URL.

## Configuration

The framework uses YAML files for corpus definitions and metric batteries.

### Corpus Configuration

A corpus config declares how to acquire and load a dataset. For example:

```yaml
name: BC5CDR

cache_filename: corpora/BC5CDR.json.gz

acquisition:
  source_url: "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip"
  format: "zip"
  converter: "bc5cdr_converter"

loader:
  name: bioc_xml
  params:
    paths:
      train: corpora/BC5CDR/CDR_TrainingSet.BioC.xml
      dev: corpora/BC5CDR/CDR_DevelopmentSet.BioC.xml
      test: corpora/BC5CDR/CDR_TestSet.BioC.xml
    label_infon_key: type
    id_infon_key: MESH
    id_format_list: [["|", "distributive", "False"]]
    nil_labels: [-1]
    default_resource: MESH
    doc_id_map:
      pmid: "__DOCUMENT_ID__"
```

If the expected loader paths already exist, the same config can be used without downloading. If the paths are missing and no `acquisition` block is present, the runner raises an explicit file-not-found error.

### Metrics Configuration

A metric battery declares corpora, dataset bundles, optional terminologies, metrics, and output paths. For example:

```yaml
corpora:
  AnatEM: configs/AnatEM.yaml
  BC5CDR: configs/BC5CDR.yaml

bundles:
  AnatEM_corpus:
    - corpus_name: AnatEM
      subset_name: train
    - corpus_name: AnatEM
      subset_name: dev
    - corpus_name: AnatEM
      subset_name: test
  BC5CDR_corpus:
    - corpus_name: BC5CDR
      subset_name: train
    - corpus_name: BC5CDR
      subset_name: dev
    - corpus_name: BC5CDR
      subset_name: test

metrics:
  - metric_name: document_count
    target_bundles: ["AnatEM_corpus", "BC5CDR_corpus"]
  - metric_name: label_distribution
    target_bundles: ["AnatEM_corpus", "BC5CDR_corpus"]

output_path: output/basic_corpus_stats.json
```

Terminology-aware batteries additionally define terminology loaders, for example `mesh_xml` for MeSH and `obo` for CL, MONDO, or ChEBI.

## Topic Profiles and Audits

CorpusBenchmarking includes three related topic-analysis workflows. They are intended to make high-level domain coverage explicit, auditable, and configurable.

* **Journal topic profiles** use NLM Catalog MeSH terms from journal metadata and map them to configured high-level journal topics. When a journal has no usable MeSH topics, configured journal-name fallbacks can be used.
* **Article topic profiles** use article-level MeSH terms from PubMed/PMC metadata. Article terms that cannot be mapped can fall back to the journal topic profile for the article's journal, with remaining unresolved mass reported as unknown.
* **Terminology topic profiles** map ontology concepts themselves to configured high-level branches, supporting terminology coverage summaries such as MeSH disease branches, MeSH chemical branches, or Cell Ontology cell categories.

The audit scripts write JSON files that expose how mappings were produced and where coverage is weak:

```bash
bash scripts/journal_topic_audit.sh
bash scripts/article_topic_audit.sh
bash scripts/terminology_topic_audit.sh
```

These audits are useful before interpreting dashboard topic heatmaps or manuscript tables because they show which terms were mapped directly, which relied on fallbacks, and which terms remain unmapped.

## High-Level Topic Configuration

High-level topic mappings are encoded as configuration files rather than hard-coded rules, to keep definitions reviewable and replaceable:

* `configs/article_topics.yaml` maps article MeSH terms to broad article-topic categories.
* `configs/journal_topics.yaml` maps NLM Catalog journal MeSH terms to broad journal-topic categories.
* `configs/journal_name_topic.json` supplies fallback topic labels for journals that lack usable NLM Catalog MeSH topics.
* `configs/MeSH_disease_mappings.yaml`, `configs/MeSH_chemical_mappings.yaml`, `configs/cell_ontology_mappings.yaml`, and `configs/MONDO_disease_mappings.yaml` map terminology concepts to high-level ontology branches for terminology coverage panels.

If a project needs a different topic taxonomy, update the mapping files and rerun the relevant script.

## Entity Scope Configuration

The dashboard entity scopes are configured in the `configs/dashboard.yaml` configuration file. Initial configuration allows each corpus to be viewed through multiple entity scopes where data are available:
* Disease
* Chemical
* Cell
* Anatomy
* Gene/protein/sequence
* Species
* Function/process scopes

## Outputs

Common outputs include:

* `basic_corpus_stats.json`: scale, density, label, ambiguity, and variation metrics.
* `overlap_stats.json`: train-test overlap and independence metrics.
* `metadata_stats.json`: journal, year, journal-topic, and article-topic metrics.
* `terminology_coverage_stats.json`: terminology coverage, ontology depth, and high-level branch coverage.
* `docs/dashboard.html`: a self-contained interactive dashboard summarizing the metrics.
* `*_audit.json`: optional topic and terminology mapping audit outputs.

## Citation

If you find this project helpful, please cite it:

```
@inproceedings{leaman_islamaj_2026,
	author={Leaman, Robert and Islamaj, Rezarta and Lu, Zhiyong}
	title={What Do Biomedical NER and Entity Linking Benchmarks Measure? A Corpus-Centric Diagnostic Framework}
	booktitle = {Proceedings of the 25th {Workshop} on {Biomedical} {Language} {Processing}},
	publisher = {Association for Computational Linguistics},
	address = {San Diego, California, USA},
	month = jul,
	year = {2026},
	pages   = {to appear}
}
```
