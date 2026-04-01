"""
shell_company_detector.py
--------------------------
Composite risk scoring for shell company and hidden ownership detection.

This module combines:
  - Graph topology analysis (cycle detection, layer depth)
  - PageRank and Betweenness centrality
  - Louvain community detection
  - Entity resolution (duplicate/alias detection)

Each entity receives a ``risk_score`` between 0.0 and 1.0 together with
a list of ``flags`` explaining what triggered the score.  Scores above
``HIGH_RISK_THRESHOLD`` (0.7) warrant immediate KYB/AML review.

Risk factors
~~~~~~~~~~~~
+-------------------------------------+----------+
| Factor                              | Weight   |
+=====================================+==========+
| Member of circular ownership cycle  | 0.40     |
| Betweenness centrality spike        | 0.20     |
| PageRank significantly above mean   | 0.15     |
| Many ownership layers (depth ≥ 3)   | 0.15     |
| Part of suspicious community        | 0.20     |
| Possible duplicate/alias entity     | 0.15     |
+-------------------------------------+----------+

Scores are capped at 1.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from .graph_builder import GraphBuilder
from .centrality import CentralityAnalyzer
from .community_detection import CommunityDetector
from .entity_resolution import EntityResolver

# Risk thresholds
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4

# Factor weights (must sum ≤ 1 each, but they stack up to 1.0)
_W_CYCLE = 0.40
_W_BETWEENNESS = 0.20
_W_PAGERANK = 0.15
_W_DEPTH = 0.15
_W_COMMUNITY = 0.20
_W_DUPLICATE = 0.15


class ShellCompanyDetector:
    """Detect shell companies and hidden ownership in an entity graph.

    Parameters
    ----------
    graph_builder:
        A :class:`~kyb_graph_analytics.GraphBuilder` instance containing
        the populated ownership graph.
    pagerank_threshold_multiplier:
        Nodes with PageRank > *mean × multiplier* are flagged.
        Default ``2.0`` (twice the mean).
    betweenness_threshold:
        Absolute betweenness centrality score above which a node is
        flagged as a structural bridge.  Default ``0.1``.
    max_community_size:
        Upper bound for ``suspicious_communities()`` in community detection.
    entity_resolution_threshold:
        Similarity threshold forwarded to :class:`~kyb_graph_analytics.EntityResolver`.
    random_state:
        Seed for Louvain; set for reproducibility.
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        pagerank_threshold_multiplier: float = 2.0,
        betweenness_threshold: float = 0.1,
        max_community_size: int = 50,
        entity_resolution_threshold: float = 0.85,
        random_state: Optional[int] = 42,
    ) -> None:
        self.gb = graph_builder
        self.graph = graph_builder.graph
        self._pr_mult = pagerank_threshold_multiplier
        self._bw_thresh = betweenness_threshold
        self._max_comm_size = max_community_size
        self._er_thresh = entity_resolution_threshold
        self._random_state = random_state

        # Sub-analysers (lazy initialised)
        self._centrality: Optional[CentralityAnalyzer] = None
        self._community: Optional[CommunityDetector] = None
        self._resolver: Optional[EntityResolver] = None

    # ------------------------------------------------------------------
    # Lazy accessor properties
    # ------------------------------------------------------------------

    @property
    def centrality(self) -> CentralityAnalyzer:
        if self._centrality is None:
            self._centrality = CentralityAnalyzer(self.graph)
        return self._centrality

    @property
    def community_detector(self) -> CommunityDetector:
        if self._community is None:
            self._community = CommunityDetector(
                self.graph, random_state=self._random_state
            )
        return self._community

    @property
    def entity_resolver(self) -> EntityResolver:
        if self._resolver is None:
            self._resolver = EntityResolver(
                self.graph, threshold=self._er_thresh
            )
        return self._resolver

    # ------------------------------------------------------------------
    # Pre-computed sets (populated in analyse())
    # ------------------------------------------------------------------

    def _build_cycle_set(self) -> set:
        """Return the set of node IDs participating in at least one cycle."""
        members: set = set()
        for cycle in self.gb.detect_cycles():
            members.update(cycle)
        return members

    def _build_suspicious_community_set(self) -> set:
        """Return the set of node IDs in suspicious communities."""
        members: set = set()
        for comm in self.community_detector.suspicious_communities(
            max_size=self._max_comm_size
        ):
            members.update(comm["members"])
        return members

    def _build_duplicate_set(self) -> set:
        """Return the set of node IDs flagged as potential duplicates."""
        members: set = set()
        for group in self.entity_resolver.duplicate_groups():
            members.update(group)
        return members

    def _ownership_depth(self, node_id: str) -> int:
        """Return the number of ownership layers above *node_id*."""
        try:
            return len(self.gb.ownership_chain(node_id))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse(self) -> List[Dict[str, Any]]:
        """Run full shell-company detection and return scored entity records.

        Returns
        -------
        list of dicts, one per graph node, with keys:
            ``"entity_id"``, ``"entity_type"``, ``"risk_score"``,
            ``"risk_level"``, ``"flags"``.

        Sorted by descending ``risk_score``.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        # Pre-compute sets and scores
        cycle_nodes = self._build_cycle_set()
        susp_community_nodes = self._build_suspicious_community_set()
        duplicate_nodes = self._build_duplicate_set()

        pr_scores = self.centrality.pagerank()
        bw_scores = self.centrality.betweenness_centrality()

        pr_mean = (
            sum(pr_scores.values()) / len(pr_scores) if pr_scores else 0.0
        )
        pr_threshold = pr_mean * self._pr_mult

        results = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            flags: List[str] = []
            score = 0.0

            # 1. Circular ownership
            if node in cycle_nodes:
                flags.append("Participates in circular ownership cycle")
                score += _W_CYCLE

            # 2. High betweenness (structural bridge)
            bw = bw_scores.get(node, 0.0)
            if bw > self._bw_thresh:
                flags.append(
                    f"High betweenness centrality ({bw:.3f} > {self._bw_thresh})"
                )
                score += _W_BETWEENNESS

            # 3. Elevated PageRank
            pr = pr_scores.get(node, 0.0)
            if pr > pr_threshold and pr_threshold > 0:
                flags.append(
                    f"PageRank ({pr:.4f}) exceeds 2× mean ({pr_mean:.4f})"
                )
                score += _W_PAGERANK

            # 4. Deep ownership chain
            depth = self._ownership_depth(node)
            if depth >= 3:
                flags.append(
                    f"Deep ownership chain ({depth} layers above this entity)"
                )
                score += _W_DEPTH

            # 5. Suspicious community membership
            if node in susp_community_nodes:
                flags.append(
                    "Member of a community with no traceable individual UBO"
                )
                score += _W_COMMUNITY

            # 6. Potential duplicate / alias
            if node in duplicate_nodes:
                flags.append(
                    "Possible duplicate or alias of another entity"
                )
                score += _W_DUPLICATE

            # Cap at 1.0
            score = min(score, 1.0)

            risk_level = (
                "high"
                if score >= HIGH_RISK_THRESHOLD
                else ("medium" if score >= MEDIUM_RISK_THRESHOLD else "low")
            )

            results.append(
                {
                    "entity_id": node,
                    "entity_type": node_data.get("entity_type", "unknown"),
                    "risk_score": round(score, 4),
                    "risk_level": risk_level,
                    "flags": flags,
                }
            )

        return sorted(results, key=lambda r: r["risk_score"], reverse=True)

    # ------------------------------------------------------------------
    # Convenience summaries
    # ------------------------------------------------------------------

    def high_risk_entities(self) -> List[Dict[str, Any]]:
        """Return only entities classified as high risk (score ≥ 0.7)."""
        return [r for r in self.analyse() if r["risk_level"] == "high"]

    def summary_report(self) -> Dict[str, Any]:
        """Return an aggregate summary of the analysis.

        Returns
        -------
        dict with keys:
            ``"total_entities"``, ``"high_risk"``, ``"medium_risk"``,
            ``"low_risk"``, ``"cycle_count"``, ``"modularity"``,
            ``"duplicate_groups"``, ``"top_risks"``.
        """
        results = self.analyse()
        graph_summary = self.gb.summary()

        high = [r for r in results if r["risk_level"] == "high"]
        medium = [r for r in results if r["risk_level"] == "medium"]
        low = [r for r in results if r["risk_level"] == "low"]

        return {
            "total_entities": len(results),
            "high_risk": len(high),
            "medium_risk": len(medium),
            "low_risk": len(low),
            "cycle_count": graph_summary["cycle_count"],
            "modularity": round(self.community_detector.modularity(), 4),
            "duplicate_groups": len(self.entity_resolver.duplicate_groups()),
            "top_risks": results[:5],
        }
