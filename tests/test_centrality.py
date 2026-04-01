"""Tests for CentralityAnalyzer."""

import pytest
import networkx as nx

from kyb_graph_analytics.graph_builder import GraphBuilder
from kyb_graph_analytics.centrality import CentralityAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def star_graph():
    """Hub-and-spoke: hub -> s1, s2, s3, s4.  Hub should have high PR."""
    gb = GraphBuilder()
    gb.add_entity("hub", entity_type="company")
    for i in range(1, 5):
        gb.add_entity(f"s{i}", entity_type="company")
        gb.add_relationship("hub", f"s{i}")
    return gb


@pytest.fixture
def chain_graph():
    """Linear chain: a -> b -> c -> d -> e.  Middle nodes have high BC."""
    gb = GraphBuilder()
    nodes = list("abcde")
    for n in nodes:
        gb.add_entity(n, entity_type="company")
    for src, tgt in zip(nodes, nodes[1:]):
        gb.add_relationship(src, tgt)
    return gb


@pytest.fixture
def empty_graph():
    return GraphBuilder()


# ---------------------------------------------------------------------------
# PageRank
# ---------------------------------------------------------------------------

class TestPageRank:
    def test_returns_all_nodes(self, star_graph):
        ca = CentralityAnalyzer(star_graph.graph)
        pr = ca.pagerank()
        assert set(pr.keys()) == set(star_graph.graph.nodes())

    def test_scores_sum_to_one(self, star_graph):
        ca = CentralityAnalyzer(star_graph.graph)
        pr = ca.pagerank()
        assert abs(sum(pr.values()) - 1.0) < 1e-4

    def test_spokes_have_higher_pr_than_hub_in_directed_star(self, star_graph):
        # In a directed star (hub -> s1..s4), spokes *receive* inbound links so
        # they accumulate more PageRank than the hub (which has no inbound edges).
        ca = CentralityAnalyzer(star_graph.graph)
        pr = ca.pagerank()
        spoke_avg = sum(pr[f"s{i}"] for i in range(1, 5)) / 4
        assert spoke_avg > pr["hub"]

    def test_empty_graph_returns_empty(self, empty_graph):
        ca = CentralityAnalyzer(empty_graph.graph)
        assert ca.pagerank() == {}


# ---------------------------------------------------------------------------
# Betweenness Centrality
# ---------------------------------------------------------------------------

class TestBetweennessCentrality:
    def test_returns_all_nodes(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        bc = ca.betweenness_centrality()
        assert set(bc.keys()) == set(chain_graph.graph.nodes())

    def test_middle_nodes_have_higher_bc(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        bc = ca.betweenness_centrality()
        # In a -> b -> c -> d -> e, 'c' is the true midpoint
        assert bc["c"] >= bc["a"]
        assert bc["c"] >= bc["e"]

    def test_all_scores_in_range(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        bc = ca.betweenness_centrality()
        for score in bc.values():
            assert 0.0 <= score <= 1.0

    def test_empty_graph_returns_empty(self, empty_graph):
        ca = CentralityAnalyzer(empty_graph.graph)
        assert ca.betweenness_centrality() == {}


# ---------------------------------------------------------------------------
# Degree Centrality
# ---------------------------------------------------------------------------

class TestDegreeCentrality:
    def test_in_degree_nonempty(self, star_graph):
        ca = CentralityAnalyzer(star_graph.graph)
        in_deg = ca.in_degree_centrality()
        assert set(in_deg.keys()) == set(star_graph.graph.nodes())
        # Spokes receive edges, hub does not
        spoke_in = in_deg["s1"]
        hub_in = in_deg["hub"]
        assert spoke_in > hub_in

    def test_out_degree_hub_highest(self, star_graph):
        ca = CentralityAnalyzer(star_graph.graph)
        out_deg = ca.out_degree_centrality()
        assert out_deg["hub"] == max(out_deg.values())

    def test_empty_graph_returns_empty(self, empty_graph):
        ca = CentralityAnalyzer(empty_graph.graph)
        assert ca.in_degree_centrality() == {}
        assert ca.out_degree_centrality() == {}


# ---------------------------------------------------------------------------
# Combined scores
# ---------------------------------------------------------------------------

class TestAllCentralityScores:
    def test_combined_keys(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        all_scores = ca.all_centrality_scores()
        for node in chain_graph.graph.nodes():
            assert node in all_scores
            assert set(all_scores[node].keys()) == {
                "pagerank", "betweenness", "in_degree", "out_degree"
            }

    def test_top_nodes(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        top = ca.top_nodes(measure="betweenness", n=3)
        assert len(top) == 3
        # Results should be sorted descending
        assert top[0][1] >= top[1][1] >= top[2][1]

    def test_top_nodes_invalid_measure(self, chain_graph):
        ca = CentralityAnalyzer(chain_graph.graph)
        with pytest.raises(ValueError, match="Unknown measure"):
            ca.top_nodes(measure="invalid")

    def test_top_nodes_capped_at_n(self, star_graph):
        ca = CentralityAnalyzer(star_graph.graph)
        top = ca.top_nodes(measure="pagerank", n=2)
        assert len(top) == 2
