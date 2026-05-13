"""
corpus_dashboard.py
Generates a self-contained HTML dashboard from corpus statistics JSON files.
Optionally incorporates train/test overlap and metadata (journal/year/topic) statistics.

Usage:
    python dashboard.py stats.json
    python dashboard.py stats.json --overlap overlap.json
    python dashboard.py stats.json --overlap overlap.json \\
                              --metadata metadata.json --output report.html --open
"""

import argparse
import json
import logging
import sys
import webbrowser
from pathlib import Path

import yaml

from corpus_benchmark.dashboard import (
    load_corpora_stats,
    load_overlaps_stats,
    attach_overlaps_to_corpora,
    load_metadata_stats,
    attach_metadata_to_corpora,
    load_terminology_stats,
    process_terminology_stats,
    attach_terminology_to_corpora,
    load_dashboard_config,
    build_html,
)

logger = logging.getLogger(__name__)
DEFAULT_DASHBOARD_CONFIG = Path("configs/dashboard.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML corpus statistics dashboard."
    )
    parser.add_argument("input", help="Corpus statistics JSON file")
    parser.add_argument(
        "--overlap",
        "-v",
        default=None,
        metavar="FILE",
        help="Optional train/test overlap statistics JSON file",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default=None,
        metavar="FILE",
        help="Optional journal/year metadata statistics JSON file",
    )
    parser.add_argument(
        "--terminology",
        "-t",
        default=None,
        metavar="FILE",
        help="Optional terminology coverage statistics JSON file",
    )
    parser.add_argument(
        "--dashboard-config",
        default=None,
        metavar="FILE",
        help="Optional dashboard configuration YAML file (default: configs/dashboard.yaml if present)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output HTML path (default: <input stem>_dashboard.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated file in the default browser",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = (
        Path(args.output)
        if args.output
        else in_path.with_name(in_path.stem + "_dashboard.html")
    )

    logger.info("Loading stats: %s", in_path)
    try:
        corpora = load_corpora_stats(str(in_path))
    except FileNotFoundError:
        logger.error("Error: file not found - %s", in_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Error: invalid JSON - %s", e)
        sys.exit(1)

    if args.overlap:
        logger.info("Loading overlap: %s", args.overlap)
        try:
            attach_overlaps_to_corpora(corpora, load_overlaps_stats(args.overlap))
            logger.info(
                "Overlap matched: %s / %s",
                sum(1 for c in corpora if c.get("overlap")),
                len(corpora),
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: overlap - %s", e)

    if args.metadata:
        logger.info("Loading metadata: %s", args.metadata)
        try:
            attach_metadata_to_corpora(corpora, load_metadata_stats(args.metadata))
            n_m = sum(
                1 for c in corpora if (c.get("metadata") or {}).get("has_metadata")
            )
            n_t = sum(1 for c in corpora if (c.get("metadata") or {}).get("topic_dist"))
            logger.info("Metadata matched: %s / %s corpora (%s with topic data)", n_m, len(corpora), n_t)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: metadata - %s", e)

    if args.terminology:
        logger.info("Loading terminology: %s", args.terminology)
        try:
            term_raw = load_terminology_stats(args.terminology)
            term_data = process_terminology_stats(term_raw)
            attach_terminology_to_corpora(corpora, term_data)
            n_t = sum(1 for c in corpora if c.get("terminology"))
            logger.info("Terminology matched: %s / %s corpora", n_t, len(corpora))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Warning: terminology - %s", e)

    dashboard_config = {}
    dashboard_config_path = Path(args.dashboard_config) if args.dashboard_config else DEFAULT_DASHBOARD_CONFIG
    if dashboard_config_path.exists():
        logger.info("Loading dashboard config: %s", dashboard_config_path)
        try:
            dashboard_config = load_dashboard_config(dashboard_config_path)
        except (OSError, yaml.YAMLError, ValueError) as e:
            logger.warning("Warning: dashboard config - %s", e)

    logger.info("Corpora: %s (%s)", len(corpora), ", ".join(c["name"] for c in corpora))
    out_path.write_text(build_html(corpora, dashboard_config), encoding="utf-8")
    logger.info("Written: %s", out_path)
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
