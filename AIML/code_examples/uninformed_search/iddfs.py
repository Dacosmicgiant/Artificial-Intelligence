# --- DLS Function (Dependency for IDDFS) ---
def dls(graph, start, goal, limit):
    """Performs Depth-Limited Search (Helper for IDDFS)."""
    def recurse(node, path, depth):
        if depth > limit:
            return None
        path = path + [node] # Create new path list for this exploration
        if node == goal:
            return path
        for neighbor in graph.get(node, []):
            # Pass the extended path (path + [node]), not the path copy logic from enhanced DLS
            result = recurse(neighbor, path, depth + 1)
            if result:
                return result
        return None
    # Note: Corrected DLS helper call path logic for clarity
    return recurse(start, [], -1) # Start depth at -1 so 'start' node is at depth 0 after increment

# Let's refine DLS slightly for better path handling and depth start
def dls_refined(graph, start, goal, limit):
    """Refined DLS: handles path correctly and starts depth appropriately."""
    def recurse(node, current_path, depth):
        # Path for this call includes the current node
        path_here = current_path + [node]
        # print(f"DLS Trace: Trying Node {node} at depth {depth}, Path: {path_here}") # Optional trace

        if depth > limit:
            # print(f"DLS Trace: Limit {limit} exceeded at depth {depth}") # Optional trace
            return None # Limit exceeded

        if node == goal:
            # print(f"DLS Trace: Goal found!") # Optional trace
            return path_here # Goal found

        if depth == limit:
            # print(f"DLS Trace: Limit {limit} reached, not goal. Stopping this branch.") # Optional trace
            return None # Reached limit without finding goal

        # Explore neighbors
        for neighbor in graph.get(node, []):
            # Avoid cycles explicitly within DLS - prevents redundant search in this specific DLS call
            # Note: Basic DLS might not include this, but it's often useful.
            # If we *don't* check for cycles here, IDDFS still works but is less efficient.
            if neighbor not in path_here: # Simple cycle check
                 result = recurse(neighbor, path_here, depth + 1)
                 if result:
                    return result # Propagate success

        # print(f"DLS Trace: No path found from {node} within limit.") # Optional trace
        return None # Goal not found from this node within limit

    # Initial call: Start node, empty path leading to it, depth 0
    return recurse(start, [], 0)


# --- IDDFS Function ---
def iddfs(graph, start, goal, max_depth):
    """
    Performs Iterative Deepening Depth-First Search.

    Args:
        graph: Adjacency list representation of the graph.
        start: The starting node.
        goal: The target node.
        max_depth: The maximum depth limit to try.
    """
    print(f"--- Starting IDDFS from '{start}' to '{goal}' with max_depth={max_depth} ---")
    # Iterate through depth limits from 0 to max_depth
    for depth in range(max_depth + 1):
        print(f"\nIDDFS: Trying DLS with limit = {depth}")
        # Call DLS with the current depth limit
        path = dls_refined(graph, start, goal, depth) # Use the refined DLS

        # If DLS found a path, return it immediately
        if path:
            print(f"IDDFS: Path found by DLS at depth {depth}: {path}")
            return path
        else:
            print(f"IDDFS: DLS with limit {depth} did not find the goal.")

    # If the loop finishes without finding the goal
    print(f"\nIDDFS: Goal '{goal}' not found within max_depth {max_depth}.")
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

start_node = 'A'
goal_node = 'F'
max_depth_limit = 5 # Example max depth

result = iddfs(graph, start_node, goal_node, max_depth_limit)
print("\n--- IDDFS Execution Finished ---")

if result:
    print(f"Final Path Found by IDDFS: {result}")
else:
    print(f"No path found from '{start_node}' to '{goal_node}' within max_depth {max_depth_limit}.")