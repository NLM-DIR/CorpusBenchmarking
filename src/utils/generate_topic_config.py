import json
import pickle
import sys
from pathlib import Path
import yaml

# Ensure src is in PYTHONPATH if needed, but assuming it's run from the project root
# or with src in PYTHONPATH.
try:
    from corpus_benchmark.models.terminologies import TerminologyResource
except ImportError:
    # Fallback for running directly from src/utils if src is not in PYTHONPATH
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from corpus_benchmark.models.terminologies import TerminologyResource

def main():
    """
    Generate a YAML topic configuration from a JSON ID-to-topic mapping.
    
    Usage: python src/utils/generate_topic_config.py <mappings.json> <terminology.pkl> <output.yaml>
    """
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <mappings.json> <terminology.pkl> <output.yaml>")
        sys.exit(1)

    mapping_path = Path(sys.argv[1])
    terminology_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        sys.exit(1)
    
    if not terminology_path.exists():
        print(f"Error: Terminology file not found: {terminology_path}")
        sys.exit(1)

    print(f"Loading mapping from {mapping_path}...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        id_to_topic = yaml.safe_load(f)

    print(f"Loading terminology from {terminology_path}...")
    with open(terminology_path, 'rb') as f:
        terminology: TerminologyResource = pickle.load(f)

    topic_to_names = {}
    missing_ids = []

    for ui, topic in id_to_topic.items():
        concept = terminology.get_concept(ui)
        if concept:
            name = concept.name
            if topic not in topic_to_names:
                topic_to_names[topic] = []
            if name not in topic_to_names[topic]:
                topic_to_names[topic].append(name)
        else:
            missing_ids.append(ui)

    if missing_ids:
        print(f"Warning: {len(missing_ids)} concepts not found in terminology:")
        for ui in missing_ids[:10]:
            print(f"  - {ui}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more.")

    # Sort names within each topic for consistency
    for topic in topic_to_names:
        topic_to_names[topic].sort()

    # Sort topics alphabetically
    sorted_topic_to_names = {topic: topic_to_names[topic] for topic in sorted(topic_to_names.keys())}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(sorted_topic_to_names, f, sort_keys=False, allow_unicode=True)

    print(f"Successfully wrote {output_path}")

if __name__ == "__main__":
    main()
