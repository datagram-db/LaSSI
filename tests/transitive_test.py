import pytest
from src.transitive_closure import floyd_warshall

def sort_adjacency_list(adj):
    for k, v in adj.items():
        adj[k] = sorted(v)

def test_small_graph1():
    adjacency_list = {
        "A": ["B"],
        "B": ["C"],
        "C": []
    }
    expected_result = {
        "A": ["B", "C"],
        "B": ["C"],
        "C": []
    }

    result = floyd_warshall(adjacency_list)
    sort_adjacency_list(result)
    assert result == expected_result


def test_small_graph2():
    adjacency_list = {
        "A": ["B", "C"],
        "B": ["A"],
        "C": ["A"]
    }
    expected_result = {
        "A": ["B", "C"],
        "B": ["A", "C"],
        "C": ["A", "B"]
    }

    result = floyd_warshall(adjacency_list)
    sort_adjacency_list(result)
    assert result == expected_result