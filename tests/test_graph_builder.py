"""Tests for GraphBuilder."""

import pytest
import networkx as nx

from kyb_graph_analytics.graph_builder import GraphBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_graph():
    """A small ownership graph: Alice -> HoldCo -> TargetCo."""
    gb = GraphBuilder()
    gb.add_entity("alice", entity_type="individual", name="Alice Smith")
    gb.add_entity("holdco", entity_type="company", name="HoldCo Ltd")
    gb.add_entity("targetco", entity_type="company", name="Target Co Ltd")
    gb.add_relationship("alice", "holdco", relationship_type="owns", weight=1.0)
    gb.add_relationship("holdco", "targetco", relationship_type="owns", weight=0.75)
    return gb


@pytest.fixture
def cyclic_graph():
    """A graph with a circular ownership cycle: A -> B -> C -> A."""
    gb = GraphBuilder()
    for node in ["A", "B", "C"]:
        gb.add_entity(node, entity_type="company")
    gb.add_relationship("A", "B")
    gb.add_relationship("B", "C")
    gb.add_relationship("C", "A")
    return gb


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

class TestAddEntity:
    def test_single_node_added(self, simple_graph):
        assert "alice" in simple_graph.graph

    def test_node_attributes(self, simple_graph):
        data = simple_graph.graph.nodes["alice"]
        assert data["entity_type"] == "individual"
        assert data["name"] == "Alice Smith"

    def test_bulk_add_entities(self):
        gb = GraphBuilder()
        entities = [
            {"id": "c1", "entity_type": "company", "name": "Corp One"},
            {"id": "c2", "entity_type": "company", "name": "Corp Two"},
        ]
        gb.add_entities(entities)
        assert gb.node_count == 2
        assert "c1" in gb.graph
        assert gb.graph.nodes["c2"]["name"] == "Corp Two"

    def test_default_entity_type(self):
        gb = GraphBuilder()
        gb.add_entity("x")
        assert gb.graph.nodes["x"]["entity_type"] == "unknown"


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

class TestAddRelationship:
    def test_edge_exists(self, simple_graph):
        assert simple_graph.graph.has_edge("alice", "holdco")
        assert simple_graph.graph.has_edge("holdco", "targetco")

    def test_edge_attributes(self, simple_graph):
        edge_data = simple_graph.graph["alice"]["holdco"]
        assert edge_data["relationship_type"] == "owns"
        assert edge_data["weight"] == 1.0

    def test_bulk_add_relationships(self):
        gb = GraphBuilder()
        gb.add_entity("a")
        gb.add_entity("b")
        gb.add_entity("c")
        gb.add_relationships([
            {"source": "a", "target": "b", "weight": 0.5},
            {"source": "b", "target": "c", "weight": 0.3},
        ])
        assert gb.edge_count == 2

    def test_from_edge_list(self):
        gb = GraphBuilder()
        gb.from_edge_list([("p1", "p2"), ("p2", "p3")])
        assert gb.node_count == 3
        assert gb.edge_count == 2


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

class TestTopologyHelpers:
    def test_ownership_chain(self, simple_graph):
        chain = simple_graph.ownership_chain("targetco")
        assert "alice" in chain
        assert "holdco" in chain
        assert "targetco" not in chain

    def test_subsidiaries(self, simple_graph):
        subs = simple_graph.subsidiaries("alice")
        assert "holdco" in subs
        assert "targetco" in subs

    def test_detect_cycles_none(self, simple_graph):
        cycles = simple_graph.detect_cycles()
        assert cycles == []

    def test_detect_cycles_present(self, cyclic_graph):
        cycles = cyclic_graph.detect_cycles()
        assert len(cycles) >= 1
        # All three nodes should appear in cycles
        cycle_nodes = {n for c in cycles for n in c}
        assert {"A", "B", "C"}.issubset(cycle_nodes)

    def test_ownership_chain_undirected_raises(self):
        gb = GraphBuilder(directed=False)
        gb.add_entity("x")
        with pytest.raises(ValueError, match="directed"):
            gb.ownership_chain("x")

    def test_subsidiaries_undirected_raises(self):
        gb = GraphBuilder(directed=False)
        gb.add_entity("x")
        with pytest.raises(ValueError, match="directed"):
            gb.subsidiaries("x")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self, simple_graph):
        s = simple_graph.summary()
        for key in ("nodes", "edges", "directed", "cycle_count", "cycles"):
            assert key in s

    def test_summary_values(self, simple_graph):
        s = simple_graph.summary()
        assert s["nodes"] == 3
        assert s["edges"] == 2
        assert s["cycle_count"] == 0
        assert s["directed"] is True

    def test_summary_cyclic(self, cyclic_graph):
        s = cyclic_graph.summary()
        assert s["cycle_count"] >= 1

    def test_subgraph(self, simple_graph):
        sg = simple_graph.get_subgraph(["alice", "holdco"])
        assert sg.number_of_nodes() == 2
        assert sg.has_edge("alice", "holdco")
