import time # To add a small delay for visualization

def dfs(graph, start, goal, visited=None, depth=0):
    """
    Performs Depth-First Search to find a path from start to goal.

    Args:
        graph: Adjacency list representation of the graph (dictionary).
        start: The starting node.
        goal: The target node.
        visited: A set of nodes already visited (used internally for recursion).
        depth: Current recursion depth (for indentation).
    """
    indent = "  " * depth # Indentation for visualizing recursion depth
    print(f"{indent}--- Entering dfs(start='{start}', goal='{goal}', visited={visited}) ---")

    # Initialize visited set on the first call
    if visited is None:
        visited = set()
        print(f"{indent}Initialized visited set: {visited}")

    # Mark the current node as visited
    visited.add(start)
    print(f"{indent}Added '{start}' to visited. Visited: {visited}")

    # --- Base Case 1: Goal Found ---
    if start == goal:
        print(f"{indent}*** Goal '{goal}' reached! Returning path: ['{start}'] ***")
        return [start] # Return the path containing just the goal node

    print(f"{indent}Exploring neighbors of '{start}': {graph[start]}")
    # --- Recursive Step: Explore Neighbors ---
    for neighbor in graph[start]:
        print(f"{indent}Checking neighbor: '{neighbor}'")
        if neighbor not in visited:
            print(f"{indent}'{neighbor}' not visited. Making recursive call: dfs('{neighbor}', '{goal}', ...)")
            time.sleep(0.5) # Small delay to make following easier
            # --- Recursive Call ---
            path_found = dfs(graph, neighbor, goal, visited.copy(), depth + 1) # Pass a copy of visited to avoid issues in other branches if backtracking occurs later

            # --- Check Result of Recursive Call ---
            if path_found:
                print(f"{indent}<<< Path found via '{neighbor}': {path_found}. Prepending '{start}'. >>>")
                # Goal was found down this path, prepend current node and return
                final_path = [start] + path_found
                print(f"{indent}Returning path: {final_path}")
                return final_path
            else:
                print(f"{indent}<<< No path to '{goal}' found via '{neighbor}'. Continuing loop. >>>")
        else:
            print(f"{indent}'{neighbor}' already visited. Skipping.")

    # --- Base Case 2: Explored all paths from 'start', goal not found ---
    print(f"{indent}--- No path found from '{start}' to '{goal}' via its unvisited neighbors. Returning None. ---")
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
goal_node = 'D'

print(f"Starting DFS from '{start_node}' to find '{goal_node}'...\n")
final_path = dfs(graph, start_node, goal_node)
print("\n--- DFS Execution Finished ---")

if final_path:
    print(f"Final Path Found: {final_path}")
else:
    print(f"No path exists between '{start_node}' and '{goal_node}'.")