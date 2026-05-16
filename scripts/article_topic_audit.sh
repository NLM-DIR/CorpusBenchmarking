set -e

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

python -m utils.ensure_nltk_data

python -m corpus_benchmark.audits.article_topic_audit \
   --output output/article_topic_audit.json \
   --topic-root-counts-output output/article_topic_root_counts.json \
   --topic-root-counts-without-fallback-output output/article_topic_root_counts_without_fallback.json \
   --unmapped-terms-output output/article_topic_unmapped_terms.json \
   --article-topics configs/article_topics.yaml \
   --journal-topics configs/journal_topics.yaml \
   --journal-name-topics configs/journal_name_topic.json
