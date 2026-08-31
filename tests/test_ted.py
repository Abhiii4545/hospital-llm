"""Zhang-Shasha reference cases.

A hand-rolled edit distance is only trustworthy if it is pinned to cases whose
answer is obvious by inspection, so these are all small and hand-countable.
"""

from __future__ import annotations

from reckon.eval.ted import (
    Node,
    json_to_tree,
    ted_accuracy,
    tree_edit_distance,
    tree_size,
)


def leaf(label: str) -> Node:
    return Node(label)


def test_identical_trees_have_zero_distance() -> None:
    a = Node("r", [leaf("x"), leaf("y")])
    b = Node("r", [leaf("x"), leaf("y")])
    assert tree_edit_distance(a, b) == 0


def test_distance_is_symmetric() -> None:
    a = Node("r", [leaf("x"), leaf("y")])
    b = Node("r", [leaf("x")])
    assert tree_edit_distance(a, b) == tree_edit_distance(b, a) == 1


def test_relabelling_the_root_costs_one() -> None:
    assert tree_edit_distance(Node("a"), Node("b")) == 1


def test_single_node_against_empty() -> None:
    assert tree_edit_distance(Node("a"), None) == 1
    assert tree_edit_distance(None, Node("a")) == 1
    assert tree_edit_distance(None, None) == 0


def test_deleting_a_subtree_costs_its_node_count() -> None:
    a = Node("r", [leaf("x"), Node("y", [leaf("y1"), leaf("y2")])])
    b = Node("r", [leaf("x")])
    assert tree_edit_distance(a, b) == 3


def test_distance_never_exceeds_total_size() -> None:
    a = Node("r", [leaf("x"), leaf("y"), leaf("z")])
    b = Node("q", [leaf("p")])
    assert tree_edit_distance(a, b) <= tree_size(a) + tree_size(b)


def test_tree_size() -> None:
    assert tree_size(None) == 0
    assert tree_size(Node("r")) == 1
    assert tree_size(Node("r", [leaf("a"), Node("b", [leaf("c")])])) == 4


def test_ted_accuracy_bounds() -> None:
    truth = Node("r", [leaf("x"), leaf("y")])
    assert ted_accuracy(truth, truth) == 1.0
    assert 0.0 <= ted_accuracy(Node("q"), truth) <= 1.0
    assert ted_accuracy(None, truth) == 0.0


def test_ted_accuracy_never_goes_negative() -> None:
    """A wildly wrong prediction floors at 0, it does not go negative."""
    truth = Node("r", [leaf("x")])
    pred = Node("q", [leaf("a"), leaf("b"), leaf("c"), leaf("d"), leaf("e")])
    assert ted_accuracy(pred, truth) == 0.0


# --------------------------------------------------------------------------
# JSON conversion
# --------------------------------------------------------------------------

def test_none_values_are_omitted() -> None:
    """Predicting null must not earn structural credit."""
    with_none = json_to_tree({"a": 1, "b": None})
    without = json_to_tree({"a": 1})
    assert tree_edit_distance(with_none, without) == 0


def test_scalar_becomes_key_and_value_nodes() -> None:
    tree = json_to_tree({"amount": "100"})
    assert tree_size(tree) == 3  # document -> amount -> "100"


def test_list_order_does_not_matter() -> None:
    a = json_to_tree({"items": [{"d": "x"}, {"d": "y"}]})
    b = json_to_tree({"items": [{"d": "y"}, {"d": "x"}]})
    assert tree_edit_distance(a, b) == 0


def test_list_content_still_matters() -> None:
    a = json_to_tree({"items": [{"d": "x"}, {"d": "y"}]})
    b = json_to_tree({"items": [{"d": "x"}, {"d": "z"}]})
    assert tree_edit_distance(a, b) == 1


def test_empty_nested_blocks_are_dropped() -> None:
    assert tree_edit_distance(
        json_to_tree({"hospital": {"name": None}, "a": 1}),
        json_to_tree({"a": 1}),
    ) == 0


def test_a_realistic_document_shape_scores_sensibly() -> None:
    truth = json_to_tree({
        "patient": {"name": "ramesh kumar", "uhid": "UH1"},
        "totals": {"net_amount": "1000"},
        "line_items": [{"description": "room rent", "amount": "500"}],
    })
    perfect = json_to_tree({
        "patient": {"name": "ramesh kumar", "uhid": "UH1"},
        "totals": {"net_amount": "1000"},
        "line_items": [{"description": "room rent", "amount": "500"}],
    })
    one_wrong = json_to_tree({
        "patient": {"name": "ramesh kumar", "uhid": "UH1"},
        "totals": {"net_amount": "9999"},
        "line_items": [{"description": "room rent", "amount": "500"}],
    })

    assert ted_accuracy(perfect, truth) == 1.0
    assert tree_edit_distance(one_wrong, truth) == 1
    assert 0.8 < ted_accuracy(one_wrong, truth) < 1.0
