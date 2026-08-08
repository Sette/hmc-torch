"""Synthetic ARFF fixtures for testing GoFun dataset loading."""

import tempfile
from pathlib import Path


def _make_arff_content(
    hierarchy_str: str, num_features: int, num_samples: int, is_go: bool = False
) -> str:
    """Build a minimal valid HMC ARFF file as a string.

    Args:
        hierarchy_str: The hierarchical class attribute value, e.g.
            ``"root.A.A1,root.A.A2,root.B"``.
        num_features: Number of numeric features.
        num_samples: Number of data rows (samples).
        is_go: If True, treat as Gene Ontology format.

    Returns:
        Full ARFF file content as a string.
    """
    lines = ["@RELATION test_hmc", ""]
    for i in range(num_features):
        lines.append(f"@ATTRIBUTE feat{i} NUMERIC")
    lines.append(f"@ATTRIBUTE class hierarchical {hierarchy_str}")
    lines.append("")
    lines.append("@DATA")

    import random

    rng = random.Random(42)
    for _ in range(num_samples):
        feats = ",".join(f"{rng.uniform(-1, 1):.4f}" for _ in range(num_features))
        # Pick one or two branches as labels
        branches = hierarchy_str.split(",")
        n_labels = rng.randint(1, min(2, len(branches)))
        chosen = rng.sample(branches, n_labels)
        label = "@".join(c.replace(".", "/") for c in chosen)
        lines.append(f"{feats},{label}")

    return "\n".join(lines)


class SyntheticARFFFixture:
    """Creates a temporary directory with synthetic ARFF files for a
    minimal GoFun-style dataset (train, valid, test)."""

    def __init__(
        self,
        name: str = "seq_FUN",
        hierarchy: str = "root.A.A1,root.A.A2,root.B",
        num_features: int = 10,
        num_train: int = 20,
        num_valid: int = 5,
        num_test: int = 5,
        is_go: bool = False,
    ):
        self.name = name
        self.hierarchy = hierarchy
        self.num_features = num_features
        self.is_go = is_go
        self._tmpdir = None
        self._counts = (num_train, num_valid, num_test)

    def _dataset_type_dir(self) -> str:
        return "datasets_GO" if self.is_go else "datasets_FUN"

    def setup(self):
        """Create the temporary ARFF files."""
        self._tmpdir = tempfile.TemporaryDirectory()
        base = Path(self._tmpdir.name) / "HMC_data_arff"
        ds_dir = base / self._dataset_type_dir() / self.name
        ds_dir.mkdir(parents=True)

        for split, n in zip(("train", "valid", "test"), self._counts):
            content = _make_arff_content(
                self.hierarchy, self.num_features, n, is_go=self.is_go
            )
            (ds_dir / f"{self.name}.{split}.arff").write_text(content, encoding="utf-8")

        return self._tmpdir.name

    def teardown(self):
        if self._tmpdir:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *_):
        self.teardown()
