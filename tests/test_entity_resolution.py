"""Tests for EntityResolver."""

import pytest
import networkx as nx

from kyb_graph_analytics.entity_resolution import (
    EntityResolver,
    _normalise,
    _token_sort_ratio,
    _lcs_length,
)


# ---------------------------------------------------------------------------
# String utility tests
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_lowercase(self):
        assert _normalise("HELLO") == "hello"

    def test_strips_accents(self):
        assert _normalise("café") == "cafe"

    def test_collapses_whitespace(self):
        assert _normalise("  a   b  ") == "a b"

    def test_removes_punctuation(self):
        assert _normalise("Ltd.") == "ltd"


class TestTokenSortRatio:
    def test_identical_strings(self):
        assert _token_sort_ratio("Alpha Corp", "Alpha Corp") == 1.0

    def test_order_invariant(self):
        s1 = _token_sort_ratio("Corp Alpha", "Alpha Corp")
        s2 = _token_sort_ratio("Alpha Corp", "Corp Alpha")
        assert s1 == s2

    def test_similar_strings(self):
        score = _token_sort_ratio("Alpha Holdings Ltd", "Alpha Holdings Limited")
        assert score > 0.8

    def test_completely_different_strings(self):
        score = _token_sort_ratio("Alpha Corp", "XYZ Ventures")
        assert score < 0.5

    def test_both_empty(self):
        assert _token_sort_ratio("", "") == 1.0

    def test_one_empty(self):
        assert _token_sort_ratio("Alpha", "") == 0.0


class TestLcsLength:
    def test_identical(self):
        assert _lcs_length("abc", "abc") == 3

    def test_no_common(self):
        assert _lcs_length("abc", "xyz") == 0

    def test_partial(self):
        assert _lcs_length("abcde", "ace") == 3


# ---------------------------------------------------------------------------
# EntityResolver fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_with_duplicates():
    """Graph containing obvious name duplicates."""
    g = nx.DiGraph()
    g.add_node("e1", entity_type="company", name="Alpha Holdings Ltd")
    g.add_node("e2", entity_type="company", name="Alpha Holdings Limited")
    g.add_node("e3", entity_type="individual", name="John Smith")
    g.add_node("e4", entity_type="individual", name="Jon Smith")
    g.add_node("e5", entity_type="company", name="Completely Different Corp")
    g.add_edge("e3", "e1")
    g.add_edge("e4", "e2")
    return g


@pytest.fixture
def graph_no_duplicates():
    g = nx.DiGraph()
    g.add_node("a", entity_type="company", name="Alpha Corp")
    g.add_node("b", entity_type="individual", name="Bob Jones")
    g.add_node("c", entity_type="company", name="Zeta Industries")
    return g


# ---------------------------------------------------------------------------
# find_duplicates
# ---------------------------------------------------------------------------

class TestFindDuplicates:
    def test_detects_near_identical_names(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.80)
        dupes = er.find_duplicates()
        pairs = {(a, b) for a, b, _ in dupes}
        assert ("e1", "e2") in pairs or ("e2", "e1") in pairs

    def test_no_false_positives_on_distinct_entities(self, graph_no_duplicates):
        er = EntityResolver(graph_no_duplicates, threshold=0.85)
        dupes = er.find_duplicates()
        assert dupes == []

    def test_scores_sorted_descending(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.70)
        dupes = er.find_duplicates()
        if len(dupes) > 1:
            for i in range(len(dupes) - 1):
                assert dupes[i][2] >= dupes[i + 1][2]

    def test_invalid_threshold_raises(self):
        g = nx.DiGraph()
        with pytest.raises(ValueError, match="threshold"):
            EntityResolver(g, threshold=1.5)


# ---------------------------------------------------------------------------
# duplicate_groups
# ---------------------------------------------------------------------------

class TestDuplicateGroups:
    def test_groups_are_lists(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.80)
        groups = er.duplicate_groups()
        assert isinstance(groups, list)
        for g in groups:
            assert isinstance(g, list)
            assert len(g) >= 2

    def test_no_groups_on_distinct_graph(self, graph_no_duplicates):
        er = EntityResolver(graph_no_duplicates, threshold=0.85)
        groups = er.duplicate_groups()
        assert groups == []


# ---------------------------------------------------------------------------
# merge_duplicates
# ---------------------------------------------------------------------------

class TestMergeDuplicates:
    def test_merged_graph_has_fewer_nodes(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.80)
        merged = er.merge_duplicates()
        assert merged.number_of_nodes() < graph_with_duplicates.number_of_nodes()

    def test_no_self_loops_after_merge(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.80)
        merged = er.merge_duplicates()
        assert list(nx.selfloop_edges(merged)) == []

    def test_merge_with_explicit_groups(self):
        g = nx.DiGraph()
        g.add_node("x", name="X Corp")
        g.add_node("y", name="Y Corp")
        g.add_node("z", name="Z Corp")
        g.add_edge("x", "z")
        g.add_edge("y", "z")
        er = EntityResolver(g, threshold=0.99)
        # Force merge x and y
        merged = er.merge_duplicates(groups=[["x", "y"]])
        # z should still exist; x and y merged to canonical
        assert "z" in merged.nodes()


# ---------------------------------------------------------------------------
# resolution_report
# ---------------------------------------------------------------------------

class TestResolutionReport:
    def test_report_has_expected_keys(self, graph_with_duplicates):
        er = EntityResolver(graph_with_duplicates, threshold=0.80)
        report = er.resolution_report()
        for item in report:
            assert "canonical" in item
            assert "aliases" in item
            assert "similarity_pairs" in item

    def test_report_empty_on_no_duplicates(self, graph_no_duplicates):
        er = EntityResolver(graph_no_duplicates, threshold=0.85)
        assert er.resolution_report() == []
