import heapq
import time # To add a small delay for visualization

def ucs(graph, start, goal):
    """
    Performs Uniform Cost Search to find the cheapest path from start to goal.

    Args:
        graph: Adjacency list representation of the weighted graph
               (dict where values are lists of (neighbor, weight) tuples).
        start: The starting node.
        goal: The target node.
    """
    print(f"--- Initializing UCS from '{start}' to '{goal}' ---")
    # Priority Queue stores tuples: (cumulative_cost, current_node, path_list_to_current_node)
    # Starts with the start node at cost 0 and an empty path leading to it.
    queue = [(0, start, [])]
    visited = set() # Stores nodes for which the cheapest path has been finalized

    print(f"Initial Priority Queue (Heap): {queue}")
    print(f"Initial Visited: {visited}")

    while queue:
        print(f"\n--- Loop Start ---")
        # Using sorted() just for printing clarity; heapq itself manages the heap property.
        print(f"Current Priority Queue (Heap View): {sorted(list(queue))}")

        # 1. Pop the element with the smallest cost from the heap
        cost, node, path = heapq.heappop(queue)
        print(f"Popped: (cost={cost}, node='{node}', path={path})")

        # 2. Visited Check: If we've already found the *cheapest* path to this node, skip
        if node in visited:
            print(f"Node '{node}' already visited with the cheapest path. Skipping.")
            continue

        # 3. Mark node as visited (finalized) and update path
        print(f"Node '{node}' not visited (or this is the cheapest path found so far).")
        # The path stored in the queue is the path *to* the current node's predecessor.
        # We add the current node now to complete the path *to* this node.
        current_path = path + [node]
        visited.add(node)
        print(f"Updated path: {current_path}")
        print(f"Visited set updated: {visited}")


        # 4. Goal Check
        if node == goal:
            print(f"*** Goal '{goal}' reached! Returning cost={cost}, path={current_path} ***")
            return (cost, current_path)

        # 5. Explore Neighbors
        print(f"Exploring neighbors of '{node}': {graph.get(node, [])}") # Use get for safety if node has no entry
        for neighbor, weight in graph.get(node, []): # Use get for nodes like 'D' or 'F'
            if neighbor not in visited:
                new_cost = cost + weight
                # Push the neighbor onto the queue with its cumulative cost.
                # The path pushed is the path *to the current node* (`current_path`).
                # The neighbor will be added to its path when *it* is popped.
                print(f"  Neighbor '{neighbor}' not visited. Pushing: (cost={new_cost}, node='{neighbor}', path={current_path})")
                heapq.heappush(queue, (new_cost, neighbor, current_path))
                time.sleep(0.5) # Small delay
            else:
                 print(f"  Neighbor '{neighbor}' already visited. Skipping push.")

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

start_node = 'A'
goal_node = 'F'

print(f"Starting UCS from '{start_node}' to find '{goal_node}'...\n")
result = ucs(weighted_graph, start_node, goal_node)
print("\n--- UCS Execution Finished ---")

if result:
    final_cost, final_path = result
    print(f"Final (Cheapest) Path Found: {final_path}")
    print(f"Total Cost: {final_cost}")
else:
    print(f"No path exists between '{start_node}' and '{goal_node}'.")