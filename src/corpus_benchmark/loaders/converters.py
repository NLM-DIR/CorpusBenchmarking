import glob
import logging
import os
import re
import shutil
from pathlib import Path

from bioc import biocxml

from corpus_benchmark.registry import register_converter
from corpus_benchmark.models.config import BenchmarkConfig

logger = logging.getLogger(__name__)

@register_converter("bc5cdr_converter")
def convert_bc5cdr(corpus_dir: Path, config: BenchmarkConfig):
    """
    BC5CDR extracts into a nested folder 'CDR_Data/CDR.Corpus.v010516/'.
    This moves the files to exactly where the loader expects them.
    """
    nested_dir = corpus_dir / "CDR_Data" / "CDR.Corpus.v010516"
    
    # Use the target paths defined in the config
    expected_paths = config.loader.params.get("paths", {})
    
    mapping = {
        "train": "CDR_TrainingSet.BioC.xml",
        "dev": "CDR_DevelopmentSet.BioC.xml",
        "test": "CDR_TestSet.BioC.xml",
    }

    for split, filename in mapping.items():
        src_path = nested_dir / filename
        target_path = Path(expected_paths[split])
        
        if src_path.exists() and not target_path.exists():
            logger.info(f"    Moving {src_path.name} to target location")
            shutil.move(src_path, target_path)
            
    # Optional cleanup of extracted temp files
    if (corpus_dir / "CDR_Data").exists():
        shutil.rmtree(corpus_dir / "CDR_Data")
    if (corpus_dir / "__MACOSX").exists():
        shutil.rmtree(corpus_dir / "__MACOSX")

@register_converter("bioid_converter")
def convert_BioID(corpus_dir: Path, config: BenchmarkConfig):
    """
    BioID extracts into a nested folder 'BioIDtraining_2/caption_bioc/'.
    The folder contains many XML files and many ._<name>.xml files
    This moves the files to exactly where the loader expects them.
    """
    nested_dir = corpus_dir / "BioIDtraining_2" / "caption_bioc"
    
    # Delete the "._*" files
    logging.debug(f"Deleting files starting with \"._\" in {nested_dir}")
    pattern = os.path.join(nested_dir, "._*")
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except OSError as e:
            logging.warning(f"Error deleting {file_path}: {e}")

    # Fix the identifiers and types
    logging.debug(f"Correcting annotation types and identifiers in {nested_dir}")
    for item in os.listdir(nested_dir):
        item_path = os.path.join(nested_dir, item)
        if os.path.isfile(item_path):
            update_bioid_identifiers(item_path)

    logging.debug("Done deleting")

def update_bioid_identifiers(filename:str):
    with open(filename, "r", encoding="utf-8") as fp:
        collection = biocxml.load(fp)
    dropped_count = 0
    for doc in collection.documents:
        for passage in doc.passages:
            annotations = []
            for annotation in passage.annotations:

                # Infer how to handle the annotation, its type and identifier
                identifier_list = annotation.infons["type"]
                type, updated_identifier_list = infer_bioid_type(identifier_list)
                if type is None:
                    dropped_count += 1
                    continue
                annotation.infons["type"] = type
                if not updated_identifier_list is None:
                    annotation.infons["identifier"] = updated_identifier_list
                annotations.append(annotation)

            passage.annotations.clear()
            annotations.sort(key=lambda x: x.locations[0].offset)
            passage.annotations.extend(annotations)
    if dropped_count > 0:
        logging.warning(f"Dropped {dropped_count} BioID annotations without a label in file \"{filename}\"")
    with open(filename, "w", encoding="utf-8") as fp:
        biocxml.dump(collection, fp)

bioid_type_handlers = {
    "BAO:": ("bioassay", True),
    "CHEBI:": ("chemical", True),
    "CL:": ("cell_type", True),
    "CVCL:": ("cell_line", True),
    "Corum:": ("gene", True), # Protein complexes
    "GO:": ("GO term", True),
    "NCBI gene:": ("gene", True),
    "NCBI taxon:": ("species", True),
    "PubChem:": ("chemical", True),
    "Rfam:": ("gene", True), # RNA families
    "Uberon:": ("anatomy", True),
    "Uniprot:": ("gene", True),
    "cell:": ("cell_type", True),
    "gene:": ("gene", True),
    "molecule:": ("chemical", True),
    "organism:": ("species", True),
    "protein:": ("gene", True),
    "subcellular:": ("anatomy", True),
    "tissue:": ("anatomy", True),
}


def update_bioid_identifier(identifier):
    identifier = re.sub(r"^Uberon:UBERON:", "Uberon:UBERON_", identifier)
    identifier = re.sub(r"^CVCL_", "CVCL:CVCL_", identifier)
    return identifier

def infer_bioid_type(identifier_list):
    identifiers = [update_bioid_identifier(id) for id in identifier_list.split("|")]
    handlers_found = set()
    for identifier in identifiers:
        found = 0
        for prefix, handler in bioid_type_handlers.items():
            if identifier.startswith(prefix):
                found += 1
                handlers_found.add(handler)
        if found != 1:
            logger.warning(f"Identifier \"{identifier}\" matched {found} handlers")
    if len(handlers_found) != 1:
        logger.warning(f"Identifier list \"{identifier_list}\" matched {len(handlers_found)} handlers")
    if len(handlers_found) == 0:
        return None, None
    handlers_found = list(handlers_found)
    handlers_found.sort()
    first = handlers_found[0]
    if first is None:
        return first, None
    type, keep_identifier = first
    if keep_identifier:
        return type, ",".join(identifiers)
    return type, None

