"""
entity_resolution.py
--------------------
Identify and merge duplicate or alias entity records in a KYB/AML graph.

Shell-company schemes frequently use slight name variations (typos,
abbreviations, transliterations) to mask that multiple records refer to
the same real-world entity.  This module provides:

* ``EntityResolver`` – fuzzy string-similarity matching that groups
  candidate duplicate entities and can collapse them in the graph.

The similarity metric is token-sort-ratio computed over the *name*
attribute of each node, falling back to the node ID when the attribute is
absent.  The implementation intentionally avoids heavy ML dependencies so
the library can run without GPU resources.
"""

from __future__ import annotations

import unicodedata
import re
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# String normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    # Strip unicode accents
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lower-case and collapse non-alphanumeric runs to single space
    cleaned = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()
    return cleaned


def _token_sort_ratio(a: str, b: str) -> float:
    """Compute a token-sort similarity ratio between two strings.

    Tokens in both strings are sorted alphabetically and joined before
    comparison, making the metric order-invariant.  Returns a float in
    [0.0, 1.0].
    """
    tokens_a = sorted(_normalise(a).split())
    tokens_b = sorted(_normalise(b).split())
    joined_a = " ".join(tokens_a)
    joined_b = " ".join(tokens_b)

    if not joined_a and not joined_b:
        return 1.0
    if not joined_a or not joined_b:
        return 0.0

    # Longest common subsequence length as similarity proxy
    lcs_len = _lcs_length(joined_a, joined_b)
    return 2 * lcs_len / (len(joined_a) + len(joined_b))


def _lcs_length(s: str, t: str) -> int:
    """Iterative LCS length computation (space-optimised)."""
    m, n = len(s), len(t)
    if m > n:
        s, t, m, n = t, s, n, m
    # Use two rows
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if t[j - 1] == s[i - 1]:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(curr[i - 1], prev[i])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


# ---------------------------------------------------------------------------
# EntityResolver
# ---------------------------------------------------------------------------

class EntityResolver:
    """Detect and optionally merge duplicate entity nodes in a graph.

    Parameters
    ----------
    graph:
        A NetworkX graph whose nodes may carry a ``"name"`` attribute used
        for similarity comparison.
    threshold:
        Minimum similarity score (0.0–1.0) to consider two entities as
        potential duplicates.  Default is ``0.85``.
    name_attr:
        Node attribute to use as the canonical name for comparison.
        Defaults to ``"name"``; falls back to the node ID when absent.
    """

    def __init__(
        self,
        graph: nx.Graph,
        threshold: float = 0.85,
        name_attr: str = "name",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.graph = graph
        self.threshold = threshold
        self.name_attr = name_attr

    # ------------------------------------------------------------------
    # Label extraction
    # ------------------------------------------------------------------

    def _label(self, node_id: str) -> str:
        """Return the comparison label for a node."""
        return str(self.graph.nodes[node_id].get(self.name_attr, node_id))

    # ------------------------------------------------------------------
    # Duplicate candidate detection
    # ------------------------------------------------------------------

    def find_duplicates(self) -> List[Tuple[str, str, float]]:
        """Return all pairs of nodes with similarity >= threshold.

        Returns
        -------
        list of ``(node_a, node_b, similarity_score)`` tuples, sorted by
        descending score.
        """
        nodes = list(self.graph.nodes())
        candidates: List[Tuple[str, str, float]] = []

        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                score = _token_sort_ratio(self._label(a), self._label(b))
                if score >= self.threshold:
                    candidates.append((a, b, score))

        return sorted(candidates, key=lambda x: x[2], reverse=True)

    def duplicate_groups(self) -> List[List[str]]:
        """Return groups of mutually similar entities using union-find.

        Returns
        -------
        list of groups, where each group is a list of node IDs that are
        considered the same real-world entity.
        """
        pairs = self.find_duplicates()
        parent: Dict[str, str] = {n: n for n in self.graph.nodes()}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for a, b, _ in pairs:
            union(a, b)

        groups: Dict[str, List[str]] = {}
        for node in self.graph.nodes():
            root = find(node)
            groups.setdefault(root, []).append(node)

        return [g for g in groups.values() if len(g) > 1]

    # ------------------------------------------------------------------
    # Graph merging
    # ------------------------------------------------------------------

    def merge_duplicates(
        self,
        groups: Optional[List[List[str]]] = None,
    ) -> nx.Graph:
        """Return a new graph where each group of duplicates is merged into
        a single canonical node.

        The canonical node for each group is the one with the longest name
        attribute (or the first alphabetically if lengths are equal).  All
        edges from/to merged nodes are redirected to the canonical node.
        Self-loops introduced by merging are removed.

        Parameters
        ----------
        groups:
            Explicit list of duplicate groups.  When *None* (default),
            :meth:`duplicate_groups` is called automatically.

        Returns
        -------
        A new NetworkX graph (same type as ``self.graph``) with duplicates
        merged.
        """
        if groups is None:
            groups = self.duplicate_groups()

        # Build a mapping: old_node → canonical_node
        merge_map: Dict[str, str] = {}
        for group in groups:
            canonical = max(group, key=lambda n: len(self._label(n)))
            for node in group:
                merge_map[node] = canonical

        # Relabel nodes in a copy of the graph
        merged = nx.relabel_nodes(self.graph, merge_map, copy=True)
        # Remove self-loops introduced by merging
        merged.remove_edges_from(list(nx.selfloop_edges(merged)))
        return merged

    # ------------------------------------------------------------------
    # Convenience report
    # ------------------------------------------------------------------

    def resolution_report(self) -> List[Dict]:
        """Return a human-readable list of detected duplicate groups.

        Returns
        -------
        list of dicts with keys:
            ``"canonical"``, ``"aliases"``, ``"similarity_pairs"``.
        """
        groups = self.duplicate_groups()
        pairs = {
            frozenset((a, b)): score
            for a, b, score in self.find_duplicates()
        }
        report = []
        for group in groups:
            canonical = max(group, key=lambda n: len(self._label(n)))
            aliases = [n for n in group if n != canonical]
            sim_pairs = [
                {"a": a, "b": b, "score": pairs[frozenset((a, b))]}
                for a in group
                for b in group
                if a < b and frozenset((a, b)) in pairs
            ]
            report.append(
                {
                    "canonical": canonical,
                    "aliases": aliases,
                    "similarity_pairs": sim_pairs,
                }
            )
        return report
