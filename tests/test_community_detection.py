"""Tests for CommunityDetector."""

import pytest
import networkx as nx

from kyb_graph_analytics.graph_builder import GraphBuilder
from kyb_graph_analytics.community_detection import CommunityDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_cluster_graph():
    """Two clearly separated cliques with a single bridging edge.

    Cluster 1: c1a, c1b, c1c (all companies)
    Cluster 2: c2a, c2b, c2c (all companies)
    Bridge:    c1c -> c2a
    """
    gb = GraphBuilder()
    for node in ["c1a", "c1b", "c1c"]:
        gb.add_entity(node, entity_type="company")
    for node in ["c2a", "c2b", "c2c"]:
        gb.add_entity(node, entity_type="company")
    # Dense intra-cluster edges
    gb.add_relationship("c1a", "c1b")
    gb.add_relationship("c1b", "c1c")
    gb.add_relationship("c1a", "c1c")
    gb.add_relationship("c2a", "c2b")
    gb.add_relationship("c2b", "c2c")
    gb.add_relationship("c2a", "c2c")
    # Bridge
    gb.add_relationship("c1c", "c2a")
    return gb.graph


@pytest.fixture
def mixed_cluster_graph():
    """A community with individuals and companies (not suspicious) plus
    one company-only community (suspicious)."""
    gb = GraphBuilder()
    # Mixed community
    gb.add_entity("alice", entity_type="individual", name="Alice")
    gb.add_entity("alpha_llc", entity_type="company", name="Alpha LLC")
    gb.add_relationship("alice", "alpha_llc")
    # Company-only community
    gb.add_entity("shell1", entity_type="company", name="Shell One")
    gb.add_entity("shell2", entity_type="company", name="Shell Two")
    gb.add_relationship("shell1", "shell2")
    return gb.graph


@pytest.fixture
def empty_graph():
    return GraphBuilder().graph


# ---------------------------------------------------------------------------
# Partition / detect
# ---------------------------------------------------------------------------

class TestDetect:
    def test_returns_partition_dict(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        partition = cd.detect()
        assert isinstance(partition, dict)
        assert set(partition.keys()) == set(two_cluster_graph.nodes())

    def test_all_values_are_ints(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        partition = cd.detect()
        assert all(isinstance(v, int) for v in partition.values())

    def test_empty_graph_returns_empty(self, empty_graph):
        cd = CommunityDetector(empty_graph)
        assert cd.detect() == {}

    def test_partition_cached_after_detect(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        p1 = cd.detect()
        assert cd.partition is p1


# ---------------------------------------------------------------------------
# Communities grouping
# ---------------------------------------------------------------------------

class TestCommunities:
    def test_returns_dict_of_lists(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        comms = cd.communities()
        assert isinstance(comms, dict)
        # All members are graph nodes
        all_members = {n for members in comms.values() for n in members}
        assert all_members == set(two_cluster_graph.nodes())

    def test_community_of_known_node(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        cd.detect()
        label = cd.community_of("c1a")
        assert isinstance(label, int)

    def test_community_of_before_detect_returns_none(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        assert cd.community_of("c1a") is None

    def test_community_of_unknown_node_returns_none(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        cd.detect()
        assert cd.community_of("nonexistent") is None


# ---------------------------------------------------------------------------
# Modularity
# ---------------------------------------------------------------------------

class TestModularity:
    def test_modularity_is_float(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        mod = cd.modularity()
        assert isinstance(mod, float)

    def test_modularity_in_valid_range(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        mod = cd.modularity()
        # Modularity for a non-degenerate partition is typically in (-1, 1)
        assert -1.0 <= mod <= 1.0

    def test_empty_graph_modularity_zero(self, empty_graph):
        cd = CommunityDetector(empty_graph)
        assert cd.modularity() == 0.0


# ---------------------------------------------------------------------------
# Suspicious communities
# ---------------------------------------------------------------------------

class TestSuspiciousCommunities:
    def test_flags_company_only_communities(self, mixed_cluster_graph):
        cd = CommunityDetector(mixed_cluster_graph)
        suspicious = cd.suspicious_communities()
        # The shell1/shell2 community should be flagged
        flagged_members = {m for c in suspicious for m in c["members"]}
        assert "shell1" in flagged_members or "shell2" in flagged_members

    def test_result_has_expected_keys(self, mixed_cluster_graph):
        cd = CommunityDetector(mixed_cluster_graph)
        suspicious = cd.suspicious_communities()
        for item in suspicious:
            assert "community_id" in item
            assert "members" in item
            assert "size" in item
            assert "reason" in item

    def test_min_size_filter(self, two_cluster_graph):
        cd = CommunityDetector(two_cluster_graph)
        # With min_size larger than total nodes, nothing is flagged
        suspicious = cd.suspicious_communities(min_size=100)
        assert suspicious == []
