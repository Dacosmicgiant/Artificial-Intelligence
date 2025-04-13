import heapq
import time # To add a small delay for visualization
from datetime import datetime # To get current time

def a_star(graph, heuristics, start, goal):
    """
    Performs A* search to find the lowest-cost path from start to goal.

    Args:
        graph: Adjacency list representation of the weighted graph.
               (dict where values are lists of (neighbor, weight) tuples).
        heuristics: Dictionary mapping nodes to their heuristic estimates (h(n)).
        start: The starting node.
        goal: The target node.
    """
    # Get current date for context
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M:%S %p %Z")
    print(f"--- Initializing A* Search from '{start}' to '{goal}' on {current_time} ---")

    # Priority Queue stores tuples: (f_value, g_value, current_node, path_list_to_current_node)
    # f_value = g_value + h_value is the priority
    g_start = 0
    h_start = heuristics.get(start, 0) # Use 0 if heuristic missing for start
    f_start = g_start + h_start
    queue = [(f_start, g_start, start, [])]
    visited = set() # Stores nodes for which the optimal path has been finalized

    print(f"Initial Priority Queue (Heap): {queue}")
    print(f"Initial Visited: {visited}")

    while queue:
        print(f"\n--- Loop Start ---")
        # Using sorted() just for printing clarity based on f-value; heapq manages the heap property.
        print(f"Current Priority Queue (Sorted by f=g+h): {sorted(list(queue))}")

        # 1. Pop the element with the smallest f-value (f = g + h) from the heap
        f_val, g_val, node, path = heapq.heappop(queue)
        print(f"Popped: (f={f_val}, g={g_val}, node='{node}', path={path})")

        # 2. Visited Check: If already finalized with optimal path, skip
        if node in visited:
            print(f"Node '{node}' already visited via optimal path. Skipping.")
            continue

        # 3. Mark node as visited (finalized) and update path
        print(f"Node '{node}' not finalized. Marking and processing.")
        visited.add(node)
        current_path = path + [node]
        print(f"Path to '{node}': {current_path} (Cost g={g_val})")
        print(f"Visited set updated: {visited}")

        # 4. Goal Check
        if node == goal:
            print(f"*** Goal '{goal}' reached! Returning cost={g_val}, path={current_path} ***")
            return (g_val, current_path) # Return actual cost (g) and path

        # 5. Explore Neighbors
        print(f"Exploring neighbors of '{node}': {graph.get(node, [])}")
        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                # Calculate cost to reach neighbor through current node
                g_neighbor = g_val + weight
                # Get heuristic for neighbor
                h_neighbor = heuristics.get(neighbor, 0) # Use 0 if missing (assuming goal heuristic is 0)
                # Calculate f-value for neighbor
                f_neighbor = g_neighbor + h_neighbor

                print(f"  Checking neighbor '{neighbor}':")
                print(f"    Cost to reach neighbor g({neighbor}) = g({node}) + cost(node, neighbor) = {g_val} + {weight} = {g_neighbor}")
                print(f"    Heuristic h({neighbor}) = {h_neighbor}")
                print(f"    Estimated total cost f({neighbor}) = g({neighbor}) + h({neighbor}) = {g_neighbor} + {h_neighbor} = {f_neighbor}")
                # Push neighbor onto queue with its f-value as priority, storing g-value and path to current node
                print(f"    Pushing: (f={f_neighbor}, g={g_neighbor}, node='{neighbor}', path={current_path})")
                heapq.heappush(queue, (f_neighbor, g_neighbor, neighbor, current_path))
                time.sleep(0.5) # Small delay
            else:
                 print(f"  Neighbor '{neighbor}' already visited via optimal path. Skipping push.")

    # 6. Queue Empty: Goal not reachable
    print(f"--- Priority Queue empty, Goal '{goal}' not found ---")
    return None

# --- The Experiment Setup ---
weighted_graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 6), ('E', 2)],
    'C': [('F', 3)],
    'D': [],
    'E': [('F', 1)],
    'F': []
}

# Heuristic values (estimated cost from node to goal 'F')
# Let's check admissibility: h(A)=7 vs actual=4 (OK), h(B)=6 vs actual=3 (OK), h(C)=5 vs actual=3 (OK), h(E)=2 vs actual=1 (OK)
heuristics = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 2, 'F': 0}

start_node = 'A'
goal_node = 'F'

print(f"Starting A* Search from '{start_node}' to '{goal_node}'...\n")
result = a_star(weighted_graph, heuristics, start_node, goal_node)
print("\n--- A* Search Execution Finished ---")

if result:
    final_cost, final_path = result
    print(f"Final (Optimal) Path Found by A*: {final_path}")
    print(f"Total Cost: {final_cost}")
else:
    print(f"No path found from '{start_node}' to '{goal_node}'.")