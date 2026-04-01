"""Tests for ShellCompanyDetector."""

import pytest

from kyb_graph_analytics.graph_builder import GraphBuilder
from kyb_graph_analytics.shell_company_detector import (
    ShellCompanyDetector,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_graph():
    """A simple, transparent ownership structure with a real UBO."""
    gb = GraphBuilder()
    gb.add_entity("alice", entity_type="individual", name="Alice Smith")
    gb.add_entity("acme", entity_type="company", name="Acme Ltd")
    gb.add_relationship("alice", "acme", weight=1.0)
    return gb


@pytest.fixture
def shell_graph():
    """A graph with multiple shell-company indicators:
    - Circular ownership: ShellA -> ShellB -> ShellC -> ShellA
    - All companies, no individual UBO
    - Deep chain for TargetCo
    """
    gb = GraphBuilder()
    for node in ["shell_a", "shell_b", "shell_c"]:
        gb.add_entity(node, entity_type="company", name=node)
    gb.add_entity("target_co", entity_type="company", name="Target Co")

    # Circular ownership cycle
    gb.add_relationship("shell_a", "shell_b")
    gb.add_relationship("shell_b", "shell_c")
    gb.add_relationship("shell_c", "shell_a")

    # Deep chain to target
    gb.add_relationship("shell_a", "target_co")
    return gb


@pytest.fixture
def empty_graph():
    return GraphBuilder()


# ---------------------------------------------------------------------------
# analyse()
# ---------------------------------------------------------------------------

class TestAnalyse:
    def test_returns_list(self, clean_graph):
        det = ShellCompanyDetector(clean_graph)
        results = det.analyse()
        assert isinstance(results, list)

    def test_all_entities_present(self, clean_graph):
        det = ShellCompanyDetector(clean_graph)
        results = det.analyse()
        ids = {r["entity_id"] for r in results}
        assert ids == set(clean_graph.graph.nodes())

    def test_result_keys(self, clean_graph):
        det = ShellCompanyDetector(clean_graph)
        for result in det.analyse():
            assert "entity_id" in result
            assert "entity_type" in result
            assert "risk_score" in result
            assert "risk_level" in result
            assert "flags" in result

    def test_sorted_by_risk_score_descending(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        results = det.analyse()
        scores = [r["risk_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_risk_score_in_range(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        for r in det.analyse():
            assert 0.0 <= r["risk_score"] <= 1.0

    def test_risk_level_matches_score(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        for r in det.analyse():
            if r["risk_score"] >= HIGH_RISK_THRESHOLD:
                assert r["risk_level"] == "high"
            elif r["risk_score"] >= MEDIUM_RISK_THRESHOLD:
                assert r["risk_level"] == "medium"
            else:
                assert r["risk_level"] == "low"

    def test_empty_graph_returns_empty(self, empty_graph):
        det = ShellCompanyDetector(empty_graph)
        assert det.analyse() == []

    def test_cycle_nodes_are_flagged(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        results = {r["entity_id"]: r for r in det.analyse()}
        # All nodes in the cycle should carry the cycle flag
        for node in ["shell_a", "shell_b", "shell_c"]:
            flags = results[node]["flags"]
            cycle_flags = [f for f in flags if "cycle" in f.lower()]
            assert len(cycle_flags) > 0

    def test_clean_graph_lower_risk_than_shell_graph(
        self, clean_graph, shell_graph
    ):
        clean_det = ShellCompanyDetector(clean_graph)
        shell_det = ShellCompanyDetector(shell_graph)
        clean_max = max(r["risk_score"] for r in clean_det.analyse())
        shell_max = max(r["risk_score"] for r in shell_det.analyse())
        assert shell_max > clean_max


# ---------------------------------------------------------------------------
# high_risk_entities()
# ---------------------------------------------------------------------------

class TestHighRiskEntities:
    def test_all_high_risk(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        high = det.high_risk_entities()
        for r in high:
            assert r["risk_score"] >= HIGH_RISK_THRESHOLD

    def test_clean_graph_no_high_risk(self, clean_graph):
        det = ShellCompanyDetector(clean_graph)
        high = det.high_risk_entities()
        # A simple two-node clean graph should produce no high-risk entities
        assert all(r["risk_score"] < HIGH_RISK_THRESHOLD for r in high)


# ---------------------------------------------------------------------------
# summary_report()
# ---------------------------------------------------------------------------

class TestSummaryReport:
    def test_summary_keys(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        summary = det.summary_report()
        for key in (
            "total_entities",
            "high_risk",
            "medium_risk",
            "low_risk",
            "cycle_count",
            "modularity",
            "duplicate_groups",
            "top_risks",
        ):
            assert key in summary

    def test_counts_sum_to_total(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        summary = det.summary_report()
        assert (
            summary["high_risk"] + summary["medium_risk"] + summary["low_risk"]
            == summary["total_entities"]
        )

    def test_cycle_count_in_shell_graph(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        summary = det.summary_report()
        assert summary["cycle_count"] >= 1

    def test_top_risks_length(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        summary = det.summary_report()
        # top_risks contains at most 5 entries
        assert len(summary["top_risks"]) <= 5

    def test_modularity_is_numeric(self, shell_graph):
        det = ShellCompanyDetector(shell_graph)
        summary = det.summary_report()
        assert isinstance(summary["modularity"], float)
