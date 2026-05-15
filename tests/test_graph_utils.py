import pytest
from utils.graph import get_topological_sort, would_create_cycle, get_leaf_nodes

def test_topological_sort_basic():
    nodes = [{'id': 0}, {'id': 1}, {'id': 2}]
    adj = {'0': ['1'], '1': ['2']}
    order = get_topological_sort(nodes, adj)
    assert order == ['0', '1', '2']

def test_topological_sort_branching():
    nodes = [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}]
    adj = {
        '0': ['1', '2'],
        '1': ['3'],
        '2': ['3']
    }
    order = get_topological_sort(nodes, adj)
    assert order[0] == '0'
    assert order[-1] == '3'
    # 1 and 2 can be in any order in between
    assert set(order[1:3]) == {'1', '2'}

def test_cycle_detection_error():
    nodes = [{'id': 0}, {'id': 1}]
    adj = {'0': ['1'], '1': ['0']}
    with pytest.raises(ValueError, match="Cycle detected"):
        get_topological_sort(nodes, adj)

def test_would_create_cycle():
    adj = {
        '0': ['1'],
        '1': ['2']
    }
    # Adding 0->2 is fine (skip connection)
    assert would_create_cycle(adj, '0', '2') == False
    # Adding 2->0 creates 0->1->2->0
    assert would_create_cycle(adj, '2', '0') == True
    # Self loop
    assert would_create_cycle(adj, '1', '1') == True

def test_leaf_nodes():
    nodes = [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}]
    adj = {
        '0': ['1', '2'],
        '1': ['3'],
        '2': []
    }
    leaves = get_leaf_nodes(nodes, adj)
    assert set(leaves) == {'2', '3'}

def test_complex_dag():
    # A more complex NAS-like cell
    nodes = [{'id': i} for i in range(5)]
    adj = {
        '0': ['1', '2', '3'],
        '1': ['4'],
        '2': ['4'],
        '3': []
    }
    order = get_topological_sort(nodes, adj)
    assert order[0] == '0'
    assert '4' in order
    assert order.index('1') < order.index('4')
    assert order.index('2') < order.index('4')
    
    leaves = get_leaf_nodes(nodes, adj)
    assert set(leaves) == {'3', '4'}
