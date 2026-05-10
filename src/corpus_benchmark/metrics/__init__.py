"""Benchmark metrics."""

import logging

logger = logging.getLogger(__name__)
logger.debug("Imported corpus_benchmark.metrics package")

# Import metrics so they register themselves.
from corpus_benchmark.metrics.basic_counts import (
    document_count,
    passage_count,
    sentence_count,
    token_count,
    mention_count,
    mention_token_count,
    identifier_count,
)
from corpus_benchmark.metrics.basic_stats import (
    passages_per_document_stats,
    annotations_per_document_stats,
    annotations_per_1000_tokens_stats,
    unique_mentions_per_document_stats,
    unique_identifiers_per_document_stats,
    spans_per_annotation_stats,
    identifiers_per_annotation_stats,
    document_length_stats,
    passage_length_stats,
    sentence_length_stats,
    token_length_stats,
    mention_length_stats,
    mention_token_length_stats,
    passage_redundancy_stats,
    sentence_redundancy_stats,
    token_redundancy_stats,
    mention_redundancy_stats,
    mention_token_redundancy_stats,
    identifier_redundancy_stats,
    variation_degree_stats,
    ambiguity_degree_stats,
)
from corpus_benchmark.metrics.unique_counts import (
    unique_tokens,
    unique_sentences,
    unique_mentions,
    unique_mention_tokens,
    unique_identifiers,
)
from corpus_benchmark.metrics.annotation_distributions import (
    label_distribution,
    identifier_resource_distribution,
    match_type_distribution,
)

from corpus_benchmark.metrics.overlaps import (
    token_overlap,
    mention_overlap,
    mention_token_overlap,
    identifier_overlap,
)

from corpus_benchmark.metrics.metadata_distribution import (
    journal_distribution,
    journal_MeSH_topic_distribution,
    journal_topic_distribution,
    publication_year_distribution,
)

from corpus_benchmark.metrics.terminology_coverage import (
    high_level_concept_counts,
    concept_depth_counts,
)

__all__ = [
    "document_count",
    "passage_count",
    "sentence_count",
    "token_count",
    "mention_count",
    "mention_token_count",
    "identifier_count",
    "passages_per_document_stats",
    "annotations_per_document_stats",
    "annotations_per_1000_tokens_stats",
    "unique_mentions_per_document_stats",
    "unique_identifiers_per_document_stats",
    "spans_per_annotation_stats",
    "identifiers_per_annotation_stats",
    "document_length_stats",
    "passage_length_stats",
    "sentence_length_stats",
    "token_length_stats",
    "mention_length_stats",
    "mention_token_length_stats",
    "passage_redundancy_stats",
    "sentence_redundancy_stats",
    "token_redundancy_stats",
    "mention_redundancy_stats",
    "mention_token_redundancy_stats",
    "identifier_redundancy_stats",
    "variation_degree_stats",
    "ambiguity_degree_stats",
    "unique_tokens",
    "unique_sentences",
    "unique_mentions",
    "unique_mention_tokens",
    "unique_identifiers",
    "label_distribution",
    "identifier_resource_distribution",
    "match_type_distribution",
    "token_overlap",
    "mention_overlap",
    "mention_token_overlap",
    "identifier_overlap",
    "journal_distribution",
    "journal_MeSH_topic_distribution",
    "journal_topic_distribution",
    "publication_year_distribution",
    "high_level_concept_counts",
    "concept_depth_counts",
]
