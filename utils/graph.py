"""
EvolveLab — Graph Utilities
Core algorithms for DAG manipulation: Kahn's Sorting, Cycle Detection, 
and Topological Integrity.
"""

from collections import deque
from typing import List, Dict, Set, Optional, Tuple

def get_topological_sort(nodes: List[Dict], adjacency: Dict[str, List[str]]) -> List[str]:
    """
    Implements Kahn's Algorithm to find a valid execution order.
    Raises ValueError if a cycle is detected.
    """
    node_ids = [str(n['id']) for n in nodes]
    in_degree = {n_id: 0 for n_id in node_ids}
    
    # Calculate in-degrees
    for src, targets in adjacency.items():
        for target in targets:
            if target in in_degree:
                in_degree[target] += 1
    
    # Queue for nodes with no incoming edges
    queue = deque([n_id for n_id in node_ids if in_degree[n_id] == 0])
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        
        # Decrement in-degree for all children
        for v in adjacency.get(u, []):
            if v in in_degree:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
    if len(topo_order) != len(node_ids):
        raise ValueError("Cycle detected in graph! Not a valid DAG.")
        
    return topo_order

def would_create_cycle(adjacency: Dict[str, List[str]], src: str, dest: str) -> bool:
    """
    Checks if adding an edge from src to dest would create a cycle.
    Uses simple DFS to see if src is reachable from dest.
    """
    if src == dest:
        return True
        
    visited = set()
    stack = [dest]
    
    while stack:
        curr = stack.pop()
        if curr == src:
            return True
        if curr not in visited:
            visited.add(curr)
            stack.extend(adjacency.get(curr, []))
            
    return False

def get_leaf_nodes(nodes: List[Dict], adjacency: Dict[str, List[str]]) -> List[str]:
    """
    Returns IDs of nodes that have no outgoing edges.
    In NAS, these are typically concatenated to form the final cell output.
    """
    all_ids = {str(n['id']) for n in nodes}
    source_ids = set(adjacency.keys())
    
    # Nodes that are not sources of any edge are leaves
    # OR nodes whose children are all outside the current node list
    leaves = []
    for n_id in all_ids:
        children = adjacency.get(n_id, [])
        # If no children in the current node set, it's a leaf
        if not any(child in all_ids for child in children):
            leaves.append(n_id)
            
    return leaves

def get_node_depths(topo_order: List[str], adjacency: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Calculates the longest path distance (depth) for each node in the DAG.
    Useful for visualization and mutation heuristics.
    """
    depths = {n_id: 0 for n_id in topo_order}
    
    for u in topo_order:
        for v in adjacency.get(u, []):
            if v in depths:
                depths[v] = max(depths[v], depths[u] + 1)
                
    return depths
