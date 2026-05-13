def normalize_issn(value: str) -> str:
    """
    Normalize ISSN values by removing hyphens and spaces.

    This stores ISSNs as eight-character strings, e.g. "1234567X".
    """

    value = value.strip().upper().replace("-", "").replace(" ", "")
    if len(value) != 8:
        raise ValueError(f"ISSN should have 8 characters after normalization: {value!r}")
    return value


def normalize_nlm_unique_id(value: str) -> str:
    """Normalize NLM catalog unique IDs for lookup."""
    value = value.strip()
    if not value:
        raise ValueError("NLM Unique ID cannot be empty.")
    return value


def normalize_doi(value: str) -> str:
    """Normalize DOI values for case-insensitive lookup."""
    return value.strip().lower()


def normalize_pmid(value: str) -> str:
    """Normalize PMID values as digit strings."""
    value = value.strip()
    if not value.isdigit():
        raise ValueError(f"PMID should contain only digits: {value!r}")
    return value


def normalize_pmcid(value: str) -> str:
    """Normalize PMCID values to PMC-prefixed uppercase strings."""
    value = value.strip().upper()
    if value.isdigit():
        value = f"PMC{value}"
    if not value.startswith("PMC"):
        raise ValueError(f"PMCID should look like PMC123456: {value!r}")
    return value
