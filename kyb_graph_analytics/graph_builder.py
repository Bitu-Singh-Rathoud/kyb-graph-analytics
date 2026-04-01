"""
graph_builder.py
----------------
Build directed, weighted ownership / relationship graphs from structured
entity data for KYB/AML investigations.

Entities represent companies, individuals, accounts, or other nodes.
Edges represent ownership, control, or transactional relationships.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


class GraphBuilder:
    """Construct and manage an ownership/relationship graph.

    Parameters
    ----------
    directed:
        When *True* (default) the graph is a ``DiGraph``; otherwise it is
        an undirected ``Graph``.  Ownership relationships are inherently
        directional, so ``directed=True`` is strongly recommended.
    """

    def __init__(self, directed: bool = True) -> None:
        self._directed = directed
        self.graph: nx.DiGraph | nx.Graph = (
            nx.DiGraph() if directed else nx.Graph()
        )

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_entity(
        self,
        entity_id: str,
        entity_type: str = "unknown",
        **attributes: Any,
    ) -> None:
        """Add a single entity node to the graph.

        Parameters
        ----------
        entity_id:
            Unique identifier for the entity (e.g. company registration
            number, person ID).
        entity_type:
            Semantic type label: ``"company"``, ``"individual"``,
            ``"account"``, etc.
        **attributes:
            Arbitrary extra node attributes (name, jurisdiction, …).
        """
        self.graph.add_node(
            entity_id,
            entity_type=entity_type,
            **attributes,
        )

    def add_entities(self, entities: Iterable[Dict[str, Any]]) -> None:
        """Bulk-add entities from an iterable of attribute dicts.

        Each dict must contain an ``"id"`` key; an optional
        ``"entity_type"`` key is recognised as a special attribute.

        Parameters
        ----------
        entities:
            Iterable of dicts, each with at minimum ``{"id": "<id>"}``.
        """
        for entity in entities:
            entity = dict(entity)
            entity_id = entity.pop("id")
            entity_type = entity.pop("entity_type", "unknown")
            self.add_entity(entity_id, entity_type=entity_type, **entity)

    # ------------------------------------------------------------------
    # Edge management
    # ------------------------------------------------------------------

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str = "owns",
        weight: float = 1.0,
        **attributes: Any,
    ) -> None:
        """Add a directed relationship edge between two entities.

        Parameters
        ----------
        source_id:
            The entity that *owns* or *controls* the target.
        target_id:
            The entity that is owned or controlled.
        relationship_type:
            Semantic label: ``"owns"``, ``"controls"``, ``"transacts"``,
            ``"directs"``, etc.
        weight:
            Ownership stake (0.0–1.0) or transaction volume.  Defaults to
            ``1.0`` (full ownership / single connection).
        **attributes:
            Extra edge attributes stored verbatim.
        """
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            weight=weight,
            **attributes,
        )

    def add_relationships(
        self, relationships: Iterable[Dict[str, Any]]
    ) -> None:
        """Bulk-add relationships from an iterable of attribute dicts.

        Each dict must contain ``"source"`` and ``"target"`` keys.
        Optional keys: ``"relationship_type"``, ``"weight"``.

        Parameters
        ----------
        relationships:
            Iterable of dicts describing edges.
        """
        for rel in relationships:
            rel = dict(rel)
            source = rel.pop("source")
            target = rel.pop("target")
            rel_type = rel.pop("relationship_type", "owns")
            weight = rel.pop("weight", 1.0)
            self.add_relationship(source, target, rel_type, weight, **rel)

    # ------------------------------------------------------------------
    # Graph-level helpers
    # ------------------------------------------------------------------

    def from_edge_list(
        self,
        edges: Iterable[Tuple[str, str]],
        relationship_type: str = "owns",
        weight: float = 1.0,
    ) -> None:
        """Populate the graph from a bare list of (source, target) tuples.

        Nodes that do not yet exist are created automatically with
        ``entity_type="unknown"``.

        Parameters
        ----------
        edges:
            Iterable of ``(source_id, target_id)`` pairs.
        relationship_type:
            Default relationship type applied to all edges.
        weight:
            Default weight applied to all edges.
        """
        for source, target in edges:
            if source not in self.graph:
                self.add_entity(source)
            if target not in self.graph:
                self.add_entity(target)
            self.add_relationship(source, target, relationship_type, weight)

    def get_subgraph(self, node_ids: List[str]) -> nx.DiGraph | nx.Graph:
        """Return a node-induced subgraph for the given entity IDs."""
        return self.graph.subgraph(node_ids).copy()

    def ownership_chain(self, entity_id: str) -> List[str]:
        """Return all ancestors of *entity_id* in the ownership hierarchy.

        In a directed graph where edges go from owner → owned, ancestors
        are the upstream owners reachable from the node via *predecessors*.

        Parameters
        ----------
        entity_id:
            The entity to trace ownership for.

        Returns
        -------
        list of str
            Ancestor entity IDs, excluding *entity_id* itself.
        """
        if not self._directed:
            raise ValueError(
                "ownership_chain() is only meaningful on a directed graph."
            )
        return [
            n
            for n in nx.ancestors(self.graph, entity_id)
            if n != entity_id
        ]

    def subsidiaries(self, entity_id: str) -> List[str]:
        """Return all descendants of *entity_id* (companies it owns).

        Parameters
        ----------
        entity_id:
            The parent entity.

        Returns
        -------
        list of str
            Descendant entity IDs.
        """
        if not self._directed:
            raise ValueError(
                "subsidiaries() is only meaningful on a directed graph."
            )
        return list(nx.descendants(self.graph, entity_id))

    def detect_cycles(self) -> List[List[str]]:
        """Return all simple cycles in the graph.

        Circular ownership (company A owns B owns C owns A) is a strong
        indicator of a shell structure.

        Returns
        -------
        list of list of str
            Each inner list is one cycle, represented as a sequence of
            node IDs.
        """
        if self._directed:
            return list(nx.simple_cycles(self.graph))
        return []

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Total number of entity nodes."""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of relationship edges."""
        return self.graph.number_of_edges()

    def summary(self) -> Dict[str, Any]:
        """Return a dict of high-level graph statistics."""
        cycles = self.detect_cycles()
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "directed": self._directed,
            "is_weakly_connected": (
                nx.is_weakly_connected(self.graph)
                if self._directed and self.node_count > 0
                else None
            ),
            "cycle_count": len(cycles),
            "cycles": cycles,
        }
