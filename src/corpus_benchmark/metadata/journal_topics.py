from __future__ import annotations

# TODO Move the journal topic classification to a representation/metadata for journals

EXACT_TOPICS = {
    "Nature": "Multidisciplinary",
    "Science": "Multidisciplinary",
    "Cell": "Cell & Dev Biology",
    "Blood": "Clinical Medicine",
    "Gut": "Clinical Medicine",
    "Pain": "Neuroscience",
    "Brain": "Neuroscience",
    "Neurology": "Neuroscience",
    "Circulation": "Cardiology & Vasc",
    "Lancet": "Clinical Medicine",
    "Langmuir": "Chemistry & Materials",
    "Small": "Chemistry & Materials",
    "Leukemia": "Oncology",
    "Stroke": "Cardiology & Vasc",
    "Diabetes": "Clinical Medicine",
    "Oncogene": "Oncology",
    "Immunity": "Immunology",
    "Neuron": "Neuroscience",
    "Bone": "Clinical Medicine",
    "Nanoscale": "Chemistry & Materials",
    "RNA": "Biochemistry & Mol Bio",
    "Structure": "Biochemistry & Mol Bio",
    "Gut Microbes": "Biochemistry & Mol Bio",
}

TOPIC_RULES = [
    (
        "Genetics & Genomics",
        ["genet", "genom", "hered", "mutat", "chromosom", "cytogenet", "bioinform"],
    ),
    (
        "Neuroscience",
        ["neuro", "brain", "psychiatr", "psychol", "epilep", "cephalalg", "alzheim"],
    ),
    (
        "Oncology",
        [
            "cancer",
            "oncol",
            "tumor",
            "tumour",
            "carcinogen",
            "anticancer",
            "leuk",
            "lymphom",
        ],
    ),
    ("Immunology", ["immunol", "allerg", "mucosal"]),
    (
        "Cell & Dev Biology",
        [
            "cell biol",
            "dev biol",
            "stem cell",
            "cell death",
            "develop",
            "embryol",
            "tissue eng",
            "reprod biol",
            "cell mol",
            "cell res",
            "cell prolif",
            "cell commun",
            "cell cycle",
            "cell tissue",
            "cell metab",
            "cell rep",
            "cell stem",
            "cell genom",
        ],
    ),
    (
        "Biochemistry & Mol Bio",
        [
            "biochem",
            "mol biol",
            "biol chem",
            "biophys",
            "febs",
            "faseb",
            "proteom",
            "glycobiol",
            "nucleic acid",
            "mol microbiol",
            "mol ecol",
            "mol endocrinol",
            "mol pharmacol",
            "mol cell",
            "mol syst",
            "mol med",
        ],
    ),
    (
        "Pharmacology & Toxicology",
        ["pharmacol", "toxicol", "xenobiotica", "antimicrob", "drug", "pharm"],
    ),
    (
        "Cardiology & Vasc",
        ["cardiol", "heart", "hypertens", "cardiovasc", "thromb", "vasc", "arteri"],
    ),
    (
        "Chemistry & Materials",
        [
            "chem",
            "nano",
            "polym",
            "colloid",
            "mater",
            "beilstein",
            "chirality",
            "ultrason",
        ],
    ),
    (
        "Clinical Medicine",
        [
            "clin",
            "intern med",
            "n engl j med",
            "jama",
            "hosp",
            "surg",
            "obstet",
            "pediatr",
            "nephrol",
            "hepatol",
            "gastroent",
            "ophthalm",
            "dermatol",
            "anesth",
            "emerg",
            "radiol",
            "urol",
            "rheumatol",
            "endocr",
            "diabet",
            "kidney",
            "liver",
            "infect",
            "transplant",
            "forensic",
            "nutr",
            "phytother",
            "planta med",
            "fitoterap",
            "vet ",
            "environ health",
        ],
    ),
    (
        "Multidisciplinary",
        [
            "plos",
            "sci rep",
            "nat commun",
            "elife",
            "proc natl acad",
            "bmc",
            "int j mol sci",
            "front ",
            "biomed res",
            "commun biol",
            "sci adv",
            "peerj",
            "gigascience",
            "sci data",
            "adv sci",
        ],
    ),
]

TOPIC_COLORS = {
    "Genetics & Genomics": "#7F77DD",
    "Pharmacology & Toxicology": "#1D9E75",
    "Multidisciplinary": "#AFA9EC",
    "Chemistry & Materials": "#D85A30",
    "Neuroscience": "#378ADD",
    "Biochemistry & Mol Bio": "#BA7517",
    "Clinical Medicine": "#5DCAA5",
    "Cell & Dev Biology": "#639922",
    "Oncology": "#E24B4A",
    "Cardiology & Vasc": "#888780",
    "Immunology": "#D4537E",
    "Other": "#D3D1C7",
}
ALL_TOPICS = list(TOPIC_COLORS.keys())


def classify_journal(name: str) -> str:
    if not name or name == "Unknown":
        return "Other"
    exact = EXACT_TOPICS.get(name)
    if exact:
        return exact
    nl = name.lower()
    for topic, keywords in TOPIC_RULES:
        if any(kw in nl for kw in keywords):
            return topic
    return "Other"


def compute_topic_dist(journal_dist: dict) -> dict:
    fracs = {t: 0.0 for t in ALL_TOPICS}
    for jname, frac in (journal_dist or {}).items():
        if jname in ("Unknown", None):
            continue
        fracs[classify_journal(jname)] += float(frac or 0)
    total = sum(fracs.values())
    if total > 0:
        return {t: round(v / total * 100, 1) for t, v in fracs.items()}
    return {t: 0.0 for t in ALL_TOPICS}
