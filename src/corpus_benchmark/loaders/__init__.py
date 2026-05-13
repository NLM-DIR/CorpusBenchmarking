"""Corpus loaders."""

import logging

logger = logging.getLogger(__name__)
logger.debug("Imported corpus_benchmark.loaders package")

# Import loaders so they register themselves.
from corpus_benchmark.loaders.bioc_loader import load_bioc_xml, load_pubtator
from corpus_benchmark.loaders.standoff_loader import load_JNLPBA_standoff, load_AnatEM_standoff
from corpus_benchmark.loaders.knowtator_loader import load_craft_knowtator
from corpus_benchmark.loaders.converters import convert_bc5cdr
from corpus_benchmark.loaders.terminology_loaders import load_mesh_xml, load_obo

__all__ = [
    "load_bioc_xml",
    "load_pubtator",
    "load_JNLPBA_standoff",
    "load_AnatEM_standoff",
    "load_craft_knowtator",
    "convert_bc5cdr",
    "load_mesh_xml",
    "load_obo",
]
