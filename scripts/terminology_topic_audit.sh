
set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m utils.ensure_nltk_data

python -m corpus_benchmark.audits.terminology_mapping_audit \
   --terminology-name cell_ontology \
   --mapping configs/cell_ontology_mappings.yaml \
   --output output/cell_ontology_mapping_audit.json

python -m corpus_benchmark.audits.terminology_mapping_audit \
   --terminology-name mesh \
   --mapping configs/MeSH_disease_mappings.yaml \
   --output output/MeSH_disease_mapping_audit.json

python -m corpus_benchmark.audits.terminology_mapping_audit \
   --terminology-name mesh \
   --mapping configs/MeSH_chemical_mappings.yaml \
   --output output/MeSH_chemical_mapping_audit.json

python -m corpus_benchmark.terminology_mapping_audit \
   --terminology-name chebi \
   --mapping configs/ChEBI_chemical_mappings.yaml \
   --output output/ChEBI_chemical_mapping_audit.json
