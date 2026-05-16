from __future__ import annotations

import sys

import nltk


REQUIRED_PACKAGES = {
    "punkt_tab": "tokenizers/punkt_tab",
}


def main() -> int:
    for package_name, resource_path in REQUIRED_PACKAGES.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading missing NLTK resource: {package_name}", file=sys.stderr)
            if not nltk.download(package_name):
                print(f"Failed to download NLTK resource: {package_name}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
