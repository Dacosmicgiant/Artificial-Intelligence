from collections import deque
import time # To add a small delay for visualization

def bfs(graph, start, goal):
    """
    Performs Breadth-First Search to find the shortest path from start to goal.

    Args:
        graph: Adjacency list representation of the graph (dictionary).
        start: The starting node.
        goal: The target node.
    """
    print(f"--- Initializing BFS from '{start}' to '{goal}' ---")
    visited = set()
    # The queue stores entire paths, starting with the path containing only the start node
    queue = deque([[start]])
    print(f"Initial Queue: {list(queue)}") # Show initial queue state
    print(f"Initial Visited: {visited}")

    while queue:
        print(f"\n--- Loop Start ---")
        print(f"Current Queue: {list(queue)}")
        # 1. Dequeue the oldest path
        path = queue.popleft()
        node = path[-1] # Get the last node in the path
        print(f"Dequeued path: {path}, Current node: '{node}'")

        # 2. Goal Check
        if node == goal:
            print(f"*** Goal '{goal}' reached! Returning path: {path} ***")
            return path

        # 3. Visited Check & Exploration
        # Process the node only if it hasn't been visited *before*
        # We add it to visited *here* to prevent adding multiple paths
        # ending in the same node to the queue later.
        if node not in visited:
            print(f"Node '{node}' not visited. Marking as visited.")
            visited.add(node)
            print(f"Visited set updated: {visited}")

            print(f"Exploring neighbors of '{node}': {graph[node]}")
            # 4. Enqueue neighbors
            for neighbor in graph[node]:
                 # Create a new path by extending the current path
                new_path = list(path)
                new_path.append(neighbor)
                print(f"  Adding new path to queue: {new_path}")
                queue.append(new_path) # Add the new path to the end of the queue
                time.sleep(0.3) # Small delay
        else:
             print(f"Node '{node}' already visited. Skipping neighbor exploration.")

    # 5. Queue Empty: Goal not reachable
    print(f"--- Queue empty, Goal '{goal}' not found ---")
    return None

# --- The Experiment Setup ---
# Using the same graph as before
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

start_node = 'A'
goal_node = 'F' # Target is 'F' this time

print(f"Starting BFS from '{start_node}' to find '{goal_node}'...\n")
final_path = bfs(graph, start_node, goal_node)
print("\n--- BFS Execution Finished ---")

if final_path:
    print(f"Final (Shortest) Path Found: {final_path}")
else:
    print(f"No path exists between '{start_node}' and '{goal_node}'.")