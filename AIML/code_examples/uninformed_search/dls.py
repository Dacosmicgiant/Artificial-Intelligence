import time # To add a small delay for visualization

# Reusing the unweighted graph from DFS/BFS examples
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

def dls(graph, start, goal, limit):
    """
    Performs Depth-Limited Search from start to goal up to a given depth limit.

    Args:
        graph: Adjacency list representation of the graph (dictionary).
        start: The starting node.
        goal: The target node.
        limit: The maximum depth to search.
    """
    print(f"--- Starting DLS from '{start}' to '{goal}' with limit={limit} ---")

    # Inner recursive function to handle the actual search
    def recurse(node, path, depth):
        indent = "  " * depth # Indentation for visualizing recursion depth
        print(f"{indent}--- Entering recurse(node='{node}', depth={depth}, path={path}) ---")

        # 1. Depth Limit Check
        print(f"{indent}Checking depth: {depth} > limit {limit}?")
        if depth > limit:
            print(f"{indent}Depth limit exceeded ({depth} > {limit}). Returning None.")
            return None

        # 2. Add current node to path (make sure to use the path passed in)
        # We modify the path list directly here. A copy is made *before* the recursive call.
        path.append(node)
        print(f"{indent}Appended '{node}'. Current path: {path}")


        # 3. Goal Check
        if node == goal:
            print(f"{indent}*** Goal '{goal}' reached at depth {depth}! Returning path: {path} ***")
            return path # Return the successful path

        # 4. Explore Neighbors (Recursive Step)
        print(f"{indent}Exploring neighbors of '{node}': {graph.get(node, [])}")
        for neighbor in graph.get(node, []): # Use get for nodes with no neighbors
            print(f"{indent}Checking neighbor: '{neighbor}'")
            time.sleep(0.5) # Small delay
            # Make a *copy* of the path before the recursive call
            # so that modifications in one branch don't affect others upon backtracking.
            print(f"{indent}Making recursive call: recurse('{neighbor}', path={path.copy()}, depth={depth + 1})")
            result = recurse(neighbor, path.copy(), depth + 1)

            # If the recursive call found the goal, propagate the result up
            if result:
                print(f"{indent}<<< Path found via '{neighbor}'. Returning result: {result} >>>")
                return result
            else:
                print(f"{indent}<<< No path to '{goal}' found via '{neighbor}' within limit. Continuing loop. >>>")


        # 5. Backtracking / No Path Found from this node within limit
        print(f"{indent}--- Explored all neighbors of '{node}' (or hit limit). Returning None from this branch. ---")
        return None

    # Initial call to the recursive function
    final_path = recurse(start, [], 0) # Start with empty path, depth 0
    print("\n--- DLS Execution Finished ---")
    return final_path

# --- The Experiment Setup ---
start_node = 'A'
goal_node = 'F'
depth_limit = 2

result = dls(graph, start_node, goal_node, depth_limit)

if result:
    print(f"Final Path Found (within limit {depth_limit}): {result}")
else:
    print(f"No path found from '{start_node}' to '{goal_node}' within depth limit {depth_limit}.")