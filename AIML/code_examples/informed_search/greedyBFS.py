import heapq
import time # To add a small delay for visualization

def best_first_search(graph, heuristics, start, goal):
    """
    Performs Greedy Best-First Search using only the heuristic value.

    Args:
        graph: Adjacency list representation of the graph.
        heuristics: Dictionary mapping nodes to their heuristic estimates (h(n)).
        start: The starting node.
        goal: The target node.
    """
    print(f"--- Initializing Greedy Best-First Search from '{start}' to '{goal}' ---")
    visited = set() # Stores nodes already visited
    # Priority Queue stores tuples: (heuristic_value, current_node, path_list_to_current_node)
    queue = [(heuristics[start], start, [])] # Start with h(start) as priority

    print(f"Initial Priority Queue (Heap): {queue}")
    print(f"Initial Visited: {visited}")

    while queue:
        print(f"\n--- Loop Start ---")
        # Using sorted() just for printing clarity based on heuristic; heapq manages the heap property.
        print(f"Current Priority Queue (Sorted by heuristic): {sorted(list(queue))}")

        # 1. Pop the element with the smallest heuristic value (h(n)) from the heap
        # The actual heuristic value is ignored after popping; it's just for ordering.
        h_value, node, path = heapq.heappop(queue)
        print(f"Popped: (h={h_value}, node='{node}', path={path})")

        # 2. Visited Check: If already visited, skip to avoid cycles/redundancy
        if node in visited:
            print(f"Node '{node}' already visited. Skipping.")
            continue

        # 3. Mark node as visited and update path
        print(f"Node '{node}' not visited. Marking and processing.")
        visited.add(node)
        current_path = path + [node]
        print(f"Path to '{node}': {current_path}")
        print(f"Visited set updated: {visited}")

        # 4. Goal Check
        if node == goal:
            print(f"*** Goal '{goal}' reached! Returning path: {current_path} ***")
            return current_path # Return the path found (might not be optimal)

        # 5. Explore Neighbors
        print(f"Exploring neighbors of '{node}': {graph.get(node, [])}")
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                 # Push neighbor onto queue, PRIORITY IS HEURISTIC OF NEIGHBOR h(neighbor)
                neighbor_h = heuristics.get(neighbor, float('inf')) # Get heuristic, use infinity if missing
                print(f"  Neighbor '{neighbor}' not visited. Pushing: (h={neighbor_h}, node='{neighbor}', path={current_path})")
                # The path stored is the path to the *current* node.
                heapq.heappush(queue, (neighbor_h, neighbor, current_path))
                time.sleep(0.5) # Small delay
            else:
                 print(f"  Neighbor '{neighbor}' already visited. Skipping push.")

    # 6. Queue Empty: Goal not reachable
    print(f"--- Priority Queue empty, Goal '{goal}' not found ---")
    return None

# --- The Experiment Setup ---
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Heuristic values (estimated cost from node to goal 'F')
heuristics = {'A': 5, 'B': 3, 'C': 4, 'D': 6, 'E': 2, 'F': 0}

start_node = 'A'
goal_node = 'F'

# Get current date for context
from datetime import datetime
current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
print(f"Search initiated on: {current_time}")


print(f"\nStarting Greedy Best-First Search from '{start_node}' to '{goal_node}'...\n")
result = best_first_search(graph, heuristics, start_node, goal_node)
print("\n--- Greedy Best-First Search Execution Finished ---")

if result:
    print(f"Final Path Found by GBFS: {result}")
    # Note: Compare this to the known shortest path A->C->F
    if result == ['A', 'C', 'F']:
        print("This path happens to be the shortest path.")
    else:
        print(f"Note: This path {result} is likely NOT the shortest path (which is ['A', 'C', 'F']). GBFS followed the heuristic greedily.")
else:
    print(f"No path found from '{start_node}' to '{goal_node}'.")