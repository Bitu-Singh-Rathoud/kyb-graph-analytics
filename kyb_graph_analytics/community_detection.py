"""
community_detection.py
----------------------
Louvain community detection for KYB/AML ownership graphs.

Communities in an ownership graph reveal clusters of closely related
entities that may constitute a single beneficial ownership group.  A
community containing many shell-like companies warrants deeper scrutiny.

Louvain is applied to the *undirected* version of the graph so that
ownership links in either direction contribute to the same community.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

try:
    import community as community_louvain  # python-louvain package
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'python-louvain' package is required for community detection. "
        "Install it with: pip install python-louvain"
    ) from exc


class CommunityDetector:
    """Detect ownership communities using the Louvain algorithm.

    Parameters
    ----------
    graph:
        A NetworkX graph (directed or undirected).  Directed graphs are
        automatically converted to undirected for community detection.
    random_state:
        Seed for the Louvain random-number generator.  Set to an integer
        for reproducible results.
    """

    def __init__(
        self,
        graph: nx.Graph,
        random_state: Optional[int] = 42,
    ) -> None:
        self.graph = graph
        self.random_state = random_state
        self._partition: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _undirected(self) -> nx.Graph:
        """Return an undirected copy of the graph, collapsing parallel edges."""
        if isinstance(self.graph, nx.DiGraph):
            return self.graph.to_undirected()
        return self.graph

    # ------------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------------

    def detect(self, resolution: float = 1.0) -> Dict[str, int]:
        """Run Louvain community detection and return the partition.

        Parameters
        ----------
        resolution:
            Controls community granularity.  Higher values produce more,
            smaller communities; lower values produce fewer, larger ones.
            Default is ``1.0`` (standard Louvain).

        Returns
        -------
        dict mapping node ID → integer community label.
        """
        undirected = self._undirected()
        if undirected.number_of_nodes() == 0:
            self._partition = {}
            return self._partition

        self._partition = community_louvain.best_partition(
            undirected,
            weight="weight",
            resolution=resolution,
            random_state=self.random_state,
        )
        return self._partition

    @property
    def partition(self) -> Optional[Dict[str, int]]:
        """The last computed partition, or *None* if :meth:`detect` has not
        been called yet."""
        return self._partition

    # ------------------------------------------------------------------
    # Community grouping
    # ------------------------------------------------------------------

    def communities(self, resolution: float = 1.0) -> Dict[int, List[str]]:
        """Return detected communities as a dict of label → member list.

        Calls :meth:`detect` internally if not already done.

        Parameters
        ----------
        resolution:
            Forwarded to :meth:`detect`.

        Returns
        -------
        dict mapping community label → list of node IDs in that community.
        """
        partition = self.detect(resolution=resolution)
        groups: Dict[int, List[str]] = {}
        for node, label in partition.items():
            groups.setdefault(label, []).append(node)
        return groups

    def community_of(self, node_id: str) -> Optional[int]:
        """Return the community label for *node_id*.

        Returns ``None`` if :meth:`detect` has not been called or the node
        does not appear in the partition.
        """
        if self._partition is None:
            return None
        return self._partition.get(node_id)

    # ------------------------------------------------------------------
    # Modularity
    # ------------------------------------------------------------------

    def modularity(self, resolution: float = 1.0) -> float:
        """Compute the modularity score of the current (or new) partition.

        Higher modularity (closer to 1.0) indicates more clearly separated
        communities; lower scores suggest poorly structured clusters.

        Parameters
        ----------
        resolution:
            Forwarded to :meth:`detect`.

        Returns
        -------
        float modularity score.
        """
        partition = self.detect(resolution=resolution)
        if not partition:
            return 0.0
        undirected = self._undirected()
        return community_louvain.modularity(partition, undirected, weight="weight")

    # ------------------------------------------------------------------
    # Suspicious community indicators
    # ------------------------------------------------------------------

    def suspicious_communities(
        self,
        min_size: int = 2,
        max_size: int = 50,
        resolution: float = 1.0,
    ) -> List[Dict]:
        """Identify communities that exhibit shell-company warning signs.

        A community is flagged as suspicious when:
        - Its size is in the range [*min_size*, *max_size*], which filters
          out trivial singletons and very large legitimate conglomerates.
        - It contains at least one entity of type ``"company"`` and at least
          one ``"individual"`` (typical ownership structure).
        - OR it consists entirely of ``"company"`` nodes with no individuals
          (layers of holding companies with no traceable UBO).

        Parameters
        ----------
        min_size:
            Minimum community size to consider.
        max_size:
            Maximum community size to consider.
        resolution:
            Forwarded to :meth:`detect`.

        Returns
        -------
        list of dicts, each with keys:
            ``"community_id"``, ``"members"``, ``"size"``,
            ``"has_individuals"``, ``"has_companies"``, ``"reason"``.
        """
        communities = self.communities(resolution=resolution)
        suspicious = []

        for label, members in communities.items():
            size = len(members)
            if size < min_size or size > max_size:
                continue

            types = [
                self.graph.nodes[m].get("entity_type", "unknown")
                for m in members
            ]
            has_individuals = any(t == "individual" for t in types)
            has_companies = any(t == "company" for t in types)
            all_companies = all(t == "company" for t in types)

            reasons = []
            if all_companies and size > 1:
                reasons.append(
                    "Community contains only company nodes with no traceable UBO"
                )
            elif has_companies and not has_individuals and size > 1:
                # Mixed companies/unknown-type entities but no individual UBO
                reasons.append(
                    "No individual beneficial owners in the ownership community"
                )

            if reasons:
                suspicious.append(
                    {
                        "community_id": label,
                        "members": members,
                        "size": size,
                        "has_individuals": has_individuals,
                        "has_companies": has_companies,
                        "reason": "; ".join(reasons),
                    }
                )

        return suspicious
