"""Unit tests for the AAPD label taxonomy and the hierarchy derived from it.

The paper reports AAPD as 98 nodes / 97 evaluated labels, which comes from
the 6 top-level fields plus the 91 sub-fields listed in ``AAPD_AREAS``.
These tests pin that shape so the taxonomy constant, the docstrings and the
paper cannot drift apart again: the canonical SGM release of AAPD is
coarser (54 labels, 6 + 48), which is where the older "6 + 48" figures in
the module docstrings and docs/expansao_dados.md came from.
"""

import pytest

from hmc.datasets.aapd.dataset_aapd import (
    AAPD_ALL_LABELS,
    AAPD_AREAS,
    AAPDHierarchyManager,
)

EXPECTED_AREAS = 6
EXPECTED_SUBFIELDS = 91
EXPECTED_LABELS = EXPECTED_AREAS + EXPECTED_SUBFIELDS
EXPECTED_NODES = EXPECTED_LABELS + 1  # 97 labels + root


class TestTaxonomyConstants:
    """The constants must describe the hierarchy that is actually reported."""

    def test_area_and_subfield_counts(self):
        assert len(AAPD_AREAS) == EXPECTED_AREAS
        assert sum(len(subs) for subs in AAPD_AREAS.values()) == EXPECTED_SUBFIELDS

    def test_total_matches_paper_node_count(self):
        """97 labels + root = 98 nodes, the AAPD row of the dataset table."""
        assert len(AAPD_ALL_LABELS) == EXPECTED_LABELS
        assert EXPECTED_NODES == 98

    def test_labels_are_unique(self):
        assert len(set(AAPD_ALL_LABELS)) == len(AAPD_ALL_LABELS)

    def test_subfields_are_prefixed_by_their_area(self):
        for area, subs in AAPD_AREAS.items():
            assert subs, f"area '{area}' has no sub-fields"
            for sub in subs:
                assert sub.split(".")[0] == area, f"'{sub}' does not belong to '{area}'"


@pytest.fixture(scope="module")
def hierarchy() -> AAPDHierarchyManager:
    return AAPDHierarchyManager.from_labels(list(AAPD_ALL_LABELS))


class TestHierarchyFromTaxonomy:
    """Deriving the graph from the labels reproduces the documented shape."""

    def test_depth_and_level_sizes(self, hierarchy):
        assert hierarchy.max_depth == 2
        assert dict(hierarchy.levels_size) == {
            0: 1,  # root
            1: EXPECTED_AREAS,
            2: EXPECTED_SUBFIELDS,
        }

    def test_node_count_and_root_index(self, hierarchy):
        assert len(hierarchy.terms) == EXPECTED_NODES
        assert hierarchy.nodes_idx["root"] == 0
        assert hierarchy.a.shape == (EXPECTED_NODES, EXPECTED_NODES)

    def test_only_areas_are_internal(self, hierarchy):
        """Only the 6 areas have children, so an inference-time R-matrix can
        only rewrite those 6 of the 97 labels: ``get_constr_out`` takes the
        max over descendants, which leaves every leaf score untouched."""
        g = hierarchy.g  # edges point child -> parent
        internal = {node for node in hierarchy.terms if g.in_degree(node) > 0}
        assert internal == {"root", *AAPD_AREAS}

    def test_labels_activate_their_ancestors(self, hierarchy):
        y_global, y_local = hierarchy.get_labels("cs.CL math.OC")
        active = {t for t, i in hierarchy.nodes_idx.items() if y_global[i] == 1.0}
        assert active == {"root", "cs", "cs.CL", "math", "math.OC"}
        assert len(y_local) == hierarchy.max_depth + 1
