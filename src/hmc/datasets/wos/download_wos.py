#!/usr/bin/env python3
"""
Download and preprocess the Web of Science (WOS) dataset for hmc-torch.

Produces the same split as HPT (Wang et al., EMNLP 2022):
  64% train / 16% dev / 20% test using np.random.seed(7) + train_test_split.

Usage:
    python -m hmc.datasets.wos.download_wos --output_dir ./data/wos

Output files:
    slot.pt                    — parent_id → set(child_ids)
    value_dict.pt              — id → label_name
    WebOfScience_train.json    — 64% of data
    WebOfScience_dev.json      — 16% of data
    WebOfScience_test.json     — 20% of data
"""

import argparse
import io
import json
import logging
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# English stopwords (same as HPT preprocess_wos.py)
_ENGLISH_STOPWORDS = [
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
]

# Label hierarchy stats (from HPT preprocess_wos.py)
_STATS = {
    "Root": {
        "CS": 0,
        "Medical": 0,
        "Civil": 0,
        "ECE": 0,
        "biochemistry": 0,
        "MAE": 0,
        "Psychology": 0,
    },
    "CS": {
        "Symbolic computation": 402,
        "Computer vision": 432,
        "Computer graphics": 412,
        "Operating systems": 380,
        "Machine learning": 398,
        "Data structures": 392,
        "network security": 445,
        "Image processing": 415,
        "Parallel computing": 443,
        "Distributed computing": 403,
        "Algorithm design": 379,
        "Computer programming": 425,
        "Relational databases": 377,
        "Software engineering": 416,
        "Bioinformatics": 365,
        "Cryptography": 387,
        "Structured Storage": 43,
    },
    "Medical": {
        "Alzheimer's Disease": 368,
        "Parkinson's Disease": 298,
        "Sprains and Strains": 142,
        "Cancer": 359,
        "Sports Injuries": 365,
        "Senior Health": 118,
        "Multiple Sclerosis": 253,
        "Hepatitis C": 288,
        "Weight Loss": 327,
        "Low Testosterone": 305,
        "Fungal Infection": 372,
        "Diabetes": 353,
        "Parenting": 343,
        "Birth Control": 335,
        "Heart Disease": 291,
        "Allergies": 357,
        "Menopause": 371,
        "Emergency Contraception": 291,
        "Skin Care": 339,
        "Myelofibrosis": 198,
        "Hypothyroidism": 315,
        "Headache": 341,
        "Overactive Bladder": 340,
        "Irritable Bowel Syndrome": 336,
        "Polycythemia Vera": 148,
        "Atrial Fibrillation": 294,
        "Smoking Cessation": 257,
        "Lymphoma": 267,
        "Asthma": 317,
        "Bipolar Disorder": 260,
        "Crohn's Disease": 198,
        "Idiopathic Pulmonary Fibrosis": 246,
        "Mental Health": 222,
        "Dementia": 237,
        "Rheumatoid Arthritis": 188,
        "Osteoporosis": 320,
        "Medicare": 255,
        "Psoriatic Arthritis": 202,
        "Addiction": 309,
        "Atopic Dermatitis": 262,
        "Digestive Health": 95,
        "Healthy Sleep": 129,
        "Anxiety": 262,
        "Psoriasis": 128,
        "Ankylosing Spondylitis": 321,
        "Children's Health": 350,
        "Stress Management": 361,
        "HIV/AIDS": 358,
        "Depression": 130,
        "Migraine": 178,
        "Osteoarthritis": 305,
        "Hereditary Angioedema": 182,
        "Kidney Health": 90,
        "Autism": 309,
        "Schizophrenia": 38,
        "Outdoor Health": 2,
    },
    "Civil": {
        "Green Building": 418,
        "Water Pollution": 446,
        "Smart Material": 363,
        "Ambient Intelligence": 410,
        "Construction Management": 412,
        "Suspension Bridge": 395,
        "Geotextile": 419,
        "Stealth Technology": 148,
        "Solar Energy": 384,
        "Remote Sensing": 384,
        "Rainwater Harvesting": 441,
        "Transparent Concrete": 3,
        "Highway Network System": 4,
        "Nano Concrete": 7,
        "Bamboo as a Building Material": 2,
        "Underwater Windmill": 1,
    },
    "ECE": {
        "Electric motor": 372,
        "Satellite radio": 148,
        "Digital control": 426,
        "Microcontroller": 413,
        "Electrical network": 392,
        "Electrical generator": 240,
        "Electricity": 447,
        "Operational amplifier": 419,
        "Analog signal processing": 407,
        "State space representation": 344,
        "Signal-flow graph": 274,
        "Electrical circuits": 375,
        "Lorentz force law": 44,
        "System identification": 417,
        "PID controller": 429,
        "Voltage law": 54,
        "Control engineering": 276,
        "Single-phase electric power": 6,
    },
    "biochemistry": {
        "Molecular biology": 746,
        "Enzymology": 576,
        "Southern blotting": 510,
        "Northern blotting": 699,
        "Human Metabolism": 622,
        "Polymerase chain reaction": 750,
        "Immunology": 652,
        "Genetics": 566,
        "Cell biology": 552,
        "DNA/RNA sequencing": 14,
    },
    "MAE": {
        "Fluid mechanics": 386,
        "Hydraulics": 402,
        "computer-aided design": 371,
        "Manufacturing engineering": 346,
        "Machine design": 420,
        "Thermodynamics": 361,
        "Materials Engineering": 289,
        "Strength of materials": 335,
        "Internal combustion engine": 387,
    },
    "Psychology": {
        "Prenatal development": 389,
        "Attention": 416,
        "Eating disorders": 387,
        "Borderline personality disorder": 376,
        "Prosocial behavior": 388,
        "False memories": 362,
        "Problem-solving": 360,
        "Prejudice": 389,
        "Antisocial personality disorder": 368,
        "Nonverbal communication": 394,
        "Leadership": 350,
        "Child abuse": 404,
        "Gender roles": 395,
        "Depression": 380,
        "Social cognition": 397,
        "Seasonal affective disorder": 365,
        "Person perception": 391,
        "Media violence": 296,
        "Schizophrenia": 335,
    },
}

_MENDELEY_WOS_URL = "https://data.mendeley.com/public-api/zip/9rw3vkcfy4/download/6"
_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "application/zip,application/octet-stream,*/*",
}


def _resolve_output_dir(output_dir: str) -> Path:
    """Return the WOS-specific output directory.

    Historically some commands passed ``--output_dir ./data``.  Keep that
    command usable, but write WOS artifacts below ``data/wos`` so loaders and
    dataset layout remain consistent.
    """
    path = Path(output_dir)
    if path.name != "wos":
        path = path / "wos"
    return path


def clean_str(string: str) -> str:
    """Tokenization/string cleaning (same as HPT preprocess_wos.py)."""
    string = string.strip().strip('"')
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


_XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _cell_index(cell_ref: str) -> int:
    index = 0
    for char in cell_ref:
        if not char.isalpha():
            break
        index = index * 26 + ord(char.upper()) - ord("A") + 1
    return index - 1


def _xlsx_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("x:v", _XLSX_NS)
    if cell.get("t") == "s":
        return (
            shared_strings[int(value.text)] if value is not None and value.text else ""
        )
    if cell.get("t") == "inlineStr":
        return "".join(text.text or "" for text in cell.findall(".//x:t", _XLSX_NS))
    return value.text if value is not None and value.text else ""


def _convert_xlsx_to_data_txt(xlsx_bytes: bytes, dest: Path) -> Path:
    """Convert the Mendeley metadata workbook to the legacy Data.txt format."""
    with zipfile.ZipFile(io.BytesIO(xlsx_bytes)) as workbook:
        shared_strings = []
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        for item in shared_root.findall("x:si", _XLSX_NS):
            shared_strings.append(
                "".join(text.text or "" for text in item.findall(".//x:t", _XLSX_NS))
            )

        with (
            workbook.open("xl/worksheets/sheet1.xml") as sheet,
            open(dest, "w", encoding="utf-8") as out,
        ):
            for _, row in ET.iterparse(sheet, events=("end",)):
                if not row.tag.endswith("}row"):
                    continue

                values = [""] * 7
                for cell in row.findall("x:c", _XLSX_NS):
                    index = _cell_index(cell.get("r", ""))
                    if 0 <= index < len(values):
                        values[index] = _xlsx_value(cell, shared_strings)

                out.write("\t".join(values) + "\n")
                row.clear()

    return dest


def _extract_from_archive(archive: zipfile.ZipFile, dest: Path, source: str) -> Path:
    data_members = [
        member
        for member in archive.namelist()
        if not member.endswith("/") and Path(member).name == "Data.txt"
    ]
    if data_members:
        member = data_members[0]
        logger.info("Extracting %s from %s", member, source)
        with archive.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        return dest

    xlsx_members = [
        member
        for member in archive.namelist()
        if not member.endswith("/") and Path(member).name == "Data.xlsx"
    ]
    if xlsx_members:
        member = xlsx_members[0]
        logger.info("Converting %s from %s", member, source)
        return _convert_xlsx_to_data_txt(archive.read(member), dest)

    zip_members = [
        member
        for member in archive.namelist()
        if not member.endswith("/") and Path(member).suffix.lower() == ".zip"
    ]
    for member in zip_members:
        logger.info("Searching nested archive %s from %s", member, source)
        with zipfile.ZipFile(io.BytesIO(archive.read(member))) as nested:
            try:
                return _extract_from_archive(nested, dest, member)
            except FileNotFoundError:
                continue

    raise FileNotFoundError(f"Data.txt or Data.xlsx not found inside {source}")


def _extract_data_txt(zip_path: Path, dest: Path) -> Path:
    """Extract or build Data.txt from a downloaded WOS archive."""
    with zipfile.ZipFile(zip_path) as archive:
        return _extract_from_archive(archive, dest, str(zip_path))


def _download_file(url: str, dest: Path) -> Path:
    """Download a file with browser-like headers."""
    import urllib.request

    request = urllib.request.Request(url, headers=_DOWNLOAD_HEADERS)
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)
    return dest


def _download_raw(output_dir: Path) -> Path:
    """Download the raw WOS archive from Mendeley or use a local Data.txt copy."""
    dest = output_dir / "Data.txt"
    if dest.exists():
        logger.info("Raw data already at %s", dest)
        return dest

    # Check if HPT already has the data
    hpt_data = Path("docs/HPT/data/WebOfScience/Meta-data/Data.txt")
    if hpt_data.exists():
        logger.info("Using HPT copy at %s", hpt_data)
        return hpt_data

    try:
        archive_path = output_dir / "wos-mendeley.zip"
        logger.info("Downloading from %s …", _MENDELEY_WOS_URL)
        _download_file(_MENDELEY_WOS_URL, archive_path)
        logger.info("Downloaded archive to %s", archive_path)
        _extract_data_txt(archive_path, dest)
        logger.info("Extracted raw data to %s", dest)
    except Exception as exc:
        logger.error(
            "Failed to download. Please place Data.txt from the WOS dataset "
            "at %s or at %s. Error: %s",
            dest,
            hpt_data,
            exc,
        )
        sys.exit(1)

    return dest


def _parse_raw(data_path: Path) -> list[dict]:
    """Parse HDLTex Data.txt into list of {doc_token, doc_label}."""
    logger.info("Parsing raw data from %s …", data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []
    label_check = {}

    for line in lines[1:]:  # Skip header
        line = line.rstrip("\n").split("\t")
        if len(line) != 7:
            continue
        sample_label = [line[3].rstrip().lstrip(), line[4].rstrip().lstrip()]
        code = f"{line[0]}-{line[1]}"

        if code in label_check:
            if sample_label[1] not in label_check[code]:
                label_check[code].append(sample_label[1])
        else:
            label_check[code] = [sample_label[1]]

        # Resolve ambiguous labels: pick the one with higher count in _STATS
        for i in label_check[code]:
            if _STATS.get(sample_label[0], {}).get(i, 0) > _STATS.get(
                sample_label[0], {}
            ).get(sample_label[1], 0):
                sample_label[1] = i
                break

        doc = clean_str(line[6])
        data.append(
            {
                "doc_token": doc,
                "doc_label": sample_label,
                "doc_topic": [],
                "doc_keyword": [],
            }
        )

    logger.info("Parsed %d documents.", len(data))
    return data


def _build_label_dict(data: list[dict]) -> tuple[dict, dict, dict]:
    """Build label_dict, value_dict, and slot from the data."""
    label_dict = {}
    hiera = defaultdict(set)

    # Assign IDs to parent labels first
    for item in data:
        parent, child = item["doc_label"]
        if parent not in label_dict:
            label_dict[parent] = len(label_dict)

    # Then assign IDs to child labels
    for item in data:
        parent, child = item["doc_label"]
        if child not in label_dict:
            label_dict[child] = len(label_dict)
        hiera[label_dict[parent]].add(label_dict[child])

    value_dict = {i: v for v, i in label_dict.items()}

    return label_dict, value_dict, dict(hiera)


def _split_and_save(
    data: list[dict],
    label_dict: dict,
    output_dir: Path,
) -> tuple[int, int, int]:
    """Split data 64/16/20 using HPT methodology and save JSONL files."""
    from sklearn.model_selection import train_test_split

    np.random.seed(7)
    n = len(data)
    ids = list(range(n))
    np.random.shuffle(ids)
    np_data = np.array(data, dtype=object)[ids]

    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)

    def _save(split_data, filename):
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                line = json.dumps(
                    {
                        "token": item["doc_token"],
                        "label": [
                            label_dict[item["doc_label"][0]],
                            label_dict[item["doc_label"][1]],
                        ],
                    }
                )
                f.write(line + "\n")
        logger.info("Saved %d records to %s", len(split_data), path)

    _save(train, "WebOfScience_train.json")
    _save(val, "WebOfScience_dev.json")
    _save(test, "WebOfScience_test.json")

    return len(train), len(val), len(test)


def main():
    parser = argparse.ArgumentParser(description="Download WOS dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/wos",
        help="Output directory for WOS files (default: ./data/wos)",
    )
    parser.add_argument(
        "--raw_data",
        type=str,
        default=None,
        help="Path to local Data.txt (skips download if provided)",
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get raw data
    if args.raw_data:
        raw_path = Path(args.raw_data)
    else:
        raw_path = _download_raw(output_dir)

    # Step 2: Parse
    data = _parse_raw(raw_path)

    # Step 3: Build label encoding
    label_dict, value_dict, slot = _build_label_dict(data)

    # Save slot.pt and value_dict.pt
    import torch

    torch.save(slot, str(output_dir / "slot.pt"))
    torch.save(value_dict, str(output_dir / "value_dict.pt"))
    logger.info(
        "Saved slot.pt (%d parents) and value_dict.pt (%d labels)",
        len(slot),
        len(value_dict),
    )

    # Step 4: Split and save JSONL
    n_train, n_val, n_test = _split_and_save(data, label_dict, output_dir)

    logger.info("Done! WOS dataset ready at %s", output_dir)
    logger.info(
        "Split: train=%d (%.1f%%), dev=%d (%.1f%%), test=%d (%.1f%%)",
        n_train,
        100 * n_train / len(data),
        n_val,
        100 * n_val / len(data),
        n_test,
        100 * n_test / len(data),
    )


if __name__ == "__main__":
    main()
