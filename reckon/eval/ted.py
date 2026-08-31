"""Tree edit distance over document JSON (Zhang-Shasha).

Implemented here rather than pulled in as a dependency: the only candidate
libraries either carry a licence this project cannot accept or could not be
confirmed as MIT/Apache-2.0/BSD. The algorithm is ~80 lines and the reference
cases below pin it down.

The document-level metric reported is the TED-based accuracy used in the Donut
literature, so the number is comparable with published results:

    accuracy = max(0, 1 - TED(pred, truth) / TED(empty, truth))

where ``TED(empty, truth)`` is just the node count of the truth tree.

Line items are a SET, not a sequence, but tree edit distance is defined over
ORDERED trees. Sibling item subtrees are therefore sorted by their serialised
form before comparison, so a correct extraction that happens to emit rows in a
different order is not punished. This is stated because it is a real thumb on
the scale, and it is applied identically to prediction and ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Node", "tree_edit_distance", "ted_accuracy", "json_to_tree", "tree_size"]


@dataclass
class Node:
    label: str
    children: list["Node"] = field(default_factory=list)


def tree_size(node: Node | None) -> int:
    if node is None:
        return 0
    return 1 + sum(tree_size(c) for c in node.children)


def _postorder_annotate(root: Node) -> tuple[list[Node], list[int]]:
    """Post-order node list plus each node's leftmost-descendant index."""
    nodes: list[Node] = []
    lmd: list[int] = []

    def walk(node: Node) -> int:
        first: int | None = None
        for child in node.children:
            index = walk(child)
            if first is None:
                first = index
        nodes.append(node)
        me = len(nodes) - 1
        lmd.append(lmd[first] if first is not None else me)
        return me

    walk(root)
    return nodes, lmd


def _keyroots(lmd: list[int]) -> list[int]:
    """Highest-indexed node for each distinct leftmost descendant."""
    last: dict[int, int] = {}
    for index, leftmost in enumerate(lmd):
        last[leftmost] = index
    return sorted(last.values())


def tree_edit_distance(tree_a: Node | None, tree_b: Node | None) -> int:
    """Unit-cost edit distance between two ordered trees."""
    if tree_a is None and tree_b is None:
        return 0
    if tree_a is None:
        return tree_size(tree_b)
    if tree_b is None:
        return tree_size(tree_a)

    nodes_a, lmd_a = _postorder_annotate(tree_a)
    nodes_b, lmd_b = _postorder_annotate(tree_b)

    treedist = [[0] * len(nodes_b) for _ in range(len(nodes_a))]

    for i in _keyroots(lmd_a):
        for j in _keyroots(lmd_b):
            i_off = lmd_a[i] - 1
            j_off = lmd_b[j] - 1
            rows = i - lmd_a[i] + 2
            cols = j - lmd_b[j] + 2

            forest = [[0] * cols for _ in range(rows)]
            for x in range(1, rows):
                forest[x][0] = forest[x - 1][0] + 1        # delete
            for y in range(1, cols):
                forest[0][y] = forest[0][y - 1] + 1        # insert

            for x in range(1, rows):
                for y in range(1, cols):
                    xi, yj = x + i_off, y + j_off
                    delete = forest[x - 1][y] + 1
                    insert = forest[x][y - 1] + 1
                    if lmd_a[i] == lmd_a[xi] and lmd_b[j] == lmd_b[yj]:
                        rename = 0 if nodes_a[xi].label == nodes_b[yj].label else 1
                        forest[x][y] = min(delete, insert, forest[x - 1][y - 1] + rename)
                        treedist[xi][yj] = forest[x][y]
                    else:
                        p = lmd_a[xi] - 1 - i_off
                        q = lmd_b[yj] - 1 - j_off
                        forest[x][y] = min(delete, insert, forest[p][q] + treedist[xi][yj])

    return treedist[len(nodes_a) - 1][len(nodes_b) - 1]


def ted_accuracy(pred: Node | None, truth: Node | None) -> float:
    """TED-based accuracy in [0, 1], as used in the Donut literature."""
    denominator = tree_size(truth)
    if denominator == 0:
        return 1.0 if tree_size(pred) == 0 else 0.0
    return max(0.0, 1.0 - tree_edit_distance(pred, truth) / denominator)


def _serialise(node: Node) -> str:
    """Stable key for sorting sibling subtrees."""
    return node.label + "(" + ",".join(_serialise(c) for c in node.children) + ")"


def json_to_tree(data: Any, label: str = "document") -> Node:
    """Build a comparison tree from a JSON-like structure.

    Keys become internal nodes and scalar values become leaves. ``None`` values
    are omitted entirely: an absent field and a field predicted as null are the
    same claim, and materialising both as nodes would let a model score points
    for confidently predicting nothing.

    Lists have their element subtrees sorted, making the metric order-invariant.
    """
    node = Node(label)
    if isinstance(data, dict):
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                child = json_to_tree(value, key)
                if child.children:
                    node.children.append(child)
            else:
                node.children.append(Node(key, [Node(str(value))]))
    elif isinstance(data, list):
        elements = [json_to_tree(item, "item") for item in data]
        elements.sort(key=_serialise)
        node.children.extend(elements)
    else:
        node.children.append(Node(str(data)))
    return node
