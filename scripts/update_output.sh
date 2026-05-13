set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# TODO convert this to run as one python program

# Calculate corpus metrics
python -u src/corpus_benchmark/cli.py configs/basic_corpus_stats.yaml 
python -u src/corpus_benchmark/cli.py configs/overlap_stats.yaml 
python -u src/corpus_benchmark/cli.py configs/metadata_stats.yaml 
python -u src/corpus_benchmark/cli.py configs/terminology_coverage.yaml

# Update dashboard
python src/corpus_benchmark/dashboard.py output/basic_corpus_stats.json --overlap output/overlap_stats.json --metadata output/metadata_stats.json --terminology output/terminology_coverage_stats.json --output output/corpus_dashboard.html
