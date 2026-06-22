#!/usr/bin/env python3
"""Convert Mendeley WOS data (Data.xlsx) to HPT-format JSONL + slot.pt + value_dict.pt.

This produces exactly the same format as HPT's preprocess_wos.py + data_wos.py.
Split: 64/16/20 using HPT methodology.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_ENGLISH_STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "don't", "should", "should've", "now", "d", "ll", "m", "o", "re",
    "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't", "didn",
    "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't",
    "haven", "haven't", "isn", "isn't", "ma", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't",
]


def clean_str(string: str) -> str:
    """Same cleaning as HPT preprocess_wos.py."""
    string = string.strip().strip('"')
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True,
                        help="Path to Data.xlsx from Mendeley WOS dataset")
    parser.add_argument("--output_dir", type=str, default="./data/wos",
                        help="Output directory")
    parser.add_argument("--no-split", action="store_true",
                        help="Save single JSONL without splitting")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import openpyxl  # pylint: disable=import-outside-toplevel

    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    ws = wb.active

    data = []
    for row in ws.iter_rows(values_only=True):
        # Skip header
        if row[0] == "Y1":
            continue
        parent_label = row[3].strip()
        child_label = row[4].strip()
        abstract = str(row[6]) if row[6] else ""
        abstract = clean_str(abstract)
        data.append({
            "doc_token": abstract,
            "doc_label": [parent_label, child_label],
        })

    logger.info("Loaded %d documents from Excel", len(data))

    # Build label dict same way as HPT data_wos.py
    label_dict = {}
    hiera = defaultdict(set)

    for item in data:
        p, c = item["doc_label"]
        if p not in label_dict:
            label_dict[p] = len(label_dict)

    for item in data:
        p, c = item["doc_label"]
        if c not in label_dict:
            label_dict[c] = len(label_dict)
        hiera[label_dict[p]].add(label_dict[c])

    value_dict = {i: v for v, i in label_dict.items()}
    slot = dict(hiera)

    import torch

    torch.save(slot, str(output_dir / "slot.pt"))
    torch.save(value_dict, str(output_dir / "value_dict.pt"))
    logger.info(
        "Saved slot.pt (%d parents) and value_dict.pt (%d labels)",
        len(slot), len(value_dict),
    )

    # Build JSONL records with int labels
    records = []
    for item in data:
        p, c = item["doc_label"]
        records.append({
            "token": item["doc_token"],
            "label": [label_dict[p], label_dict[c]],
        })

    if args.no_split:
        path = output_dir / "WebOfScience_total.json"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %d records to %s", len(records), path)
        return

    # Split 64/16/20 using HPT methodology
    from sklearn.model_selection import train_test_split

    np.random.seed(7)
    n = len(records)
    ids = list(range(n))
    np.random.shuffle(ids)
    np_data = np.array(records, dtype=object)[ids]

    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)

    def _save(split_data, filename):
        path = output_dir / filename
        with open(path, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        logger.info("Saved %d records to %s", len(split_data), path)

    _save(train, "WebOfScience_train.json")
    _save(val, "WebOfScience_dev.json")
    _save(test, "WebOfScience_test.json")

    logger.info(
        "Done! Split: train=%d (%.1f%%), dev=%d (%.1f%%), test=%d (%.1f%%)",
        len(train), 100 * len(train) / n,
        len(val), 100 * len(val) / n,
        len(test), 100 * len(test) / n,
    )


if __name__ == "__main__":
    main()
