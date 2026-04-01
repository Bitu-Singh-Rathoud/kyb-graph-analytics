"""
centrality.py
-------------
Compute centrality measures on a KYB/AML ownership graph.

PageRank
    Identifies the most *influential* entities in an ownership network.
    High-PageRank nodes are likely ultimate beneficial owners (UBOs) or
    pivotal holding companies.

Betweenness Centrality
    Identifies *bridge* entities that sit on many shortest paths.
    High-betweenness nodes are often intermediary shell companies used to
    obfuscate ownership chains.

In-Degree / Out-Degree
    Simple counts of incoming/outgoing ownership edges.  An entity with
    many owners but few or no subsidiaries may be an opaque vehicle.
"""

from __future__ import annotations

from typing import Dict, Optional

import networkx as nx


class CentralityAnalyzer:
    """Compute and expose centrality metrics for an ownership graph.

    Parameters
    ----------
    graph:
        A ``networkx.DiGraph`` (or ``Graph``) representing the ownership
        network produced by :class:`~kyb_graph_analytics.GraphBuilder`.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph

    # ------------------------------------------------------------------
    # PageRank
    # ------------------------------------------------------------------

    def pagerank(
        self,
        alpha: float = 0.85,
        weight: Optional[str] = "weight",
        max_iter: int = 100,
        tol: float = 1.0e-6,
    ) -> Dict[str, float]:
        """Compute PageRank for all nodes.

        Parameters
        ----------
        alpha:
            Damping factor (default 0.85).
        weight:
            Edge attribute to use as weight.  Pass ``None`` to treat all
            edges equally.
        max_iter:
            Maximum number of iterations.
        tol:
            Convergence tolerance.

        Returns
        -------
        dict mapping node ID → PageRank score (float in [0, 1]).
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(
            self.graph,
            alpha=alpha,
            weight=weight,
            max_iter=max_iter,
            tol=tol,
        )

    # ------------------------------------------------------------------
    # Betweenness Centrality
    # ------------------------------------------------------------------

    def betweenness_centrality(
        self,
        normalized: bool = True,
        weight: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute Betweenness Centrality for all nodes.

        Parameters
        ----------
        normalized:
            When *True* (default) values are normalised to [0, 1].
        weight:
            Edge attribute interpreted as *distance* (lower weight = shorter
            path).  Pass ``None`` to count hops only.

        Returns
        -------
        dict mapping node ID → betweenness score (float in [0, 1]).
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.betweenness_centrality(
            self.graph,
            normalized=normalized,
            weight=weight,
        )

    # ------------------------------------------------------------------
    # Degree Centrality
    # ------------------------------------------------------------------

    def in_degree_centrality(self) -> Dict[str, float]:
        """Normalised in-degree centrality (for directed graphs).

        Returns
        -------
        dict mapping node ID → normalised in-degree score.
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        if isinstance(self.graph, nx.DiGraph):
            return nx.in_degree_centrality(self.graph)
        return nx.degree_centrality(self.graph)

    def out_degree_centrality(self) -> Dict[str, float]:
        """Normalised out-degree centrality (for directed graphs).

        Returns
        -------
        dict mapping node ID → normalised out-degree score.
        """
        if self.graph.number_of_nodes() == 0:
            return {}
        if isinstance(self.graph, nx.DiGraph):
            return nx.out_degree_centrality(self.graph)
        return nx.degree_centrality(self.graph)

    # ------------------------------------------------------------------
    # Combined report
    # ------------------------------------------------------------------

    def all_centrality_scores(
        self,
        pagerank_alpha: float = 0.85,
    ) -> Dict[str, Dict[str, float]]:
        """Return a combined dict of all centrality measures per node.

        Parameters
        ----------
        pagerank_alpha:
            Damping factor forwarded to :meth:`pagerank`.

        Returns
        -------
        dict mapping node ID → ``{"pagerank": …, "betweenness": …,
        "in_degree": …, "out_degree": …}``.
        """
        pr = self.pagerank(alpha=pagerank_alpha)
        bc = self.betweenness_centrality()
        in_deg = self.in_degree_centrality()
        out_deg = self.out_degree_centrality()

        return {
            node: {
                "pagerank": pr.get(node, 0.0),
                "betweenness": bc.get(node, 0.0),
                "in_degree": in_deg.get(node, 0.0),
                "out_degree": out_deg.get(node, 0.0),
            }
            for node in self.graph.nodes()
        }

    def top_nodes(
        self,
        measure: str = "pagerank",
        n: int = 10,
        pagerank_alpha: float = 0.85,
    ) -> list:
        """Return the top-*n* nodes ranked by *measure*.

        Parameters
        ----------
        measure:
            One of ``"pagerank"``, ``"betweenness"``, ``"in_degree"``,
            ``"out_degree"``.
        n:
            Number of top nodes to return.
        pagerank_alpha:
            Damping factor forwarded to :meth:`pagerank`.

        Returns
        -------
        list of ``(node_id, score)`` tuples, sorted descending.
        """
        scores_map = {
            "pagerank": self.pagerank(alpha=pagerank_alpha),
            "betweenness": self.betweenness_centrality(),
            "in_degree": self.in_degree_centrality(),
            "out_degree": self.out_degree_centrality(),
        }
        if measure not in scores_map:
            raise ValueError(
                f"Unknown measure '{measure}'. Choose from: "
                + ", ".join(scores_map)
            )
        scores = scores_map[measure]
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
