from copy import deepcopy

import pytest

from raft.graph import build_adjacency, expand_neighbors


def edge(source, target, weight):
    return {"source": source, "target": target, "weight": weight}


def test_zero_weight_undirected_edges_and_integer_string_id_distinction():
    adjacency = build_adjacency([edge("1", 1, 0), edge(1, 1, 1)])
    assert expand_neighbors([1], adjacency, per_case_top_k=1) == [
        {"seed_id": 1, "neighbors": [{"id": "1", "weight": 0}]}
    ]


def test_duplicate_edges_keep_max_weight_and_inputs_are_preserved():
    seeds = ["seed"]
    adjacency = build_adjacency(
        [edge("seed", "a", 0.7), edge("a", "seed", 0.1), edge("seed", "b", 0.6)]
    )
    before = deepcopy((seeds, adjacency))
    assert expand_neighbors(seeds, adjacency, per_case_top_k=1) == [
        {"seed_id": "seed", "neighbors": [{"id": "a", "weight": 0.7}]}
    ]
    assert (seeds, adjacency) == before


@pytest.mark.parametrize("weight", [-1, 2, float("nan"), float("inf")])
def test_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match="weights"):
        build_adjacency([edge("a", "b", weight)])


@pytest.mark.parametrize("per_case_top_k", [0, -1, True, 1.5])
def test_public_expansion_validates_per_case_limit(per_case_top_k):
    with pytest.raises(ValueError, match="per_case_top_k"):
        expand_neighbors([], {}, per_case_top_k=per_case_top_k)


def test_expansion_groups_distinct_seeds_preserving_shared_neighbors_and_empty_groups():
    graph = build_adjacency(
        [
            edge("a", "x", 0.9),
            edge("b", "x", 0.8),
            edge("a", "y", 0.6),
            edge("b", "y", 0.7),
            edge("a", "b", 1),
            edge("a", "z", 0),
        ]
    )
    result = expand_neighbors(["a", "unknown", "b", "a"], graph, per_case_top_k=3)
    assert result == [
        {
            "seed_id": "a",
            "neighbors": [
                {"id": "x", "weight": 0.9},
                {"id": "y", "weight": 0.6},
                {"id": "z", "weight": 0},
            ],
        },
        {"seed_id": "unknown", "neighbors": []},
        {
            "seed_id": "b",
            "neighbors": [{"id": "x", "weight": 0.8}, {"id": "y", "weight": 0.7}],
        },
    ]
    assert expand_neighbors(["a", "b"], graph, per_case_top_k=1, allowed_ids={"y"}) == [
        {"seed_id": "a", "neighbors": [{"id": "y", "weight": 0.6}]},
        {"seed_id": "b", "neighbors": [{"id": "y", "weight": 0.7}]},
    ]
    assert expand_neighbors([], graph) == []
    assert expand_neighbors(["unknown"], graph) == [{"seed_id": "unknown", "neighbors": []}]
    assert expand_neighbors(["a"], graph, allowed_ids=[]) == [{"seed_id": "a", "neighbors": []}]


def test_limit_applies_to_each_seed_without_an_overall_cap():
    graph = build_adjacency(
        [
            edge("a", "b", 1),
            edge("a", "x", 0.9),
            edge("a", "y", 0.8),
            edge("a", "excluded-a", 0.7),
            edge("b", "z", 0.6),
            edge("b", "w", 0.5),
            edge("b", "excluded-b", 0.4),
        ]
    )
    assert expand_neighbors(["b", "a"], graph, per_case_top_k=2) == [
        {
            "seed_id": "b",
            "neighbors": [{"id": "z", "weight": 0.6}, {"id": "w", "weight": 0.5}],
        },
        {
            "seed_id": "a",
            "neighbors": [{"id": "x", "weight": 0.9}, {"id": "y", "weight": 0.8}],
        },
    ]


def test_shared_neighbors_keep_their_own_seed_weights_within_each_seed_limit():
    graph = build_adjacency(
        [edge("a", "y", 1), edge("a", "x", 0.9), edge("b", "x", 0.5)]
    )
    assert expand_neighbors(["a", "b"], graph, per_case_top_k=1) == [
        {"seed_id": "a", "neighbors": [{"id": "y", "weight": 1}]},
        {"seed_id": "b", "neighbors": [{"id": "x", "weight": 0.5}]},
    ]
    assert expand_neighbors(["a", "b"], graph, per_case_top_k=2) == [
        {
            "seed_id": "a",
            "neighbors": [{"id": "y", "weight": 1}, {"id": "x", "weight": 0.9}],
        },
        {"seed_id": "b", "neighbors": [{"id": "x", "weight": 0.5}]},
    ]


def test_allowed_ids_filter_before_each_seed_limit():
    graph = build_adjacency(
        [
            edge("a", "blocked", 1),
            edge("b", "blocked", 0.9),
            edge("a", "x", 0.6),
            edge("b", "y", 0.5),
        ]
    )
    assert expand_neighbors(
        ["a", "b"], graph, per_case_top_k=1, allowed_ids={"x", "y"}
    ) == [
        {"seed_id": "a", "neighbors": [{"id": "x", "weight": 0.6}]},
        {"seed_id": "b", "neighbors": [{"id": "y", "weight": 0.5}]},
    ]


def test_ties_use_seed_order_and_stable_neighbor_ids():
    edges = [
        edge("b", "z", 0.5), edge("b", "y", 0.5),
        edge("a", "x", 0.5), edge("a", "w", 0.5),
    ]
    expected = [
        {"seed_id": "b", "neighbors": [{"id": "y", "weight": 0.5}]},
        {"seed_id": "a", "neighbors": [{"id": "w", "weight": 0.5}]},
    ]
    assert expand_neighbors(["b", "a"], build_adjacency(edges), per_case_top_k=1) == expected
    assert expand_neighbors(["b", "a"], build_adjacency(edges[::-1]), per_case_top_k=1) == expected


def test_expand_ids_stable_ties_and_original_id_types():
    graph = build_adjacency([edge("seed", 1, 0.5), edge("seed", "1", 0.5)])
    result = expand_neighbors(["seed"], graph)
    assert {type(n["id"]) for n in result[0]["neighbors"]} == {str, int}
    assert expand_neighbors(["seed"], graph, allowed_ids=[1]) == [
        {"seed_id": "seed", "neighbors": [{"id": 1, "weight": 0.5}]},
    ]
    assert result == expand_neighbors(
        ["seed"],
        build_adjacency(
            [
                edge("seed", "1", 0.5),
                edge("seed", 1, 0.5),
            ]
        ),
    )
