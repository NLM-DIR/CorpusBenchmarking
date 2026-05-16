set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m utils.ensure_nltk_data

python -m corpus_benchmark.audits.journal_topic_audit --output output/journal_topic_audit.json  --mesh-root-counts-output output/journal_mesh_root_counts.json --mesh-term-root-frequencies-output output/journal_mesh_term_root_frequencies.json --journal-topics configs/journal_topics.yaml --journal-name-topics configs/journal_name_topic.json
