import numpy as np
import pickle
# --------------------------------
from collections import deque
import heapq
import math
# -------------------------------
# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]
# --------------------------------------------
def dfs_limited(node, goal, depth, visited):
        if node == goal:
            return [node]
        if depth == 0:
            return None
        visited.add(node)
        for neighbor, connected in enumerate(adj_matrix[node]):
            if connected and neighbor not in visited:
                path = dfs_limited(neighbor, goal, depth - 1, visited)
                if path:
                    return [node] + path
        visited.remove(node)  # Backtrack by removing the node from visited
        return None

def is_same_component(start, goal):
        visited = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node == goal:
                return True
            if node not in visited:
                visited.add(node)
                for neighbor, connected in enumerate(adj_matrix[node]):
                    if connected and neighbor not in visited:
                        stack.append(neighbor)
        return False
# ----------------------------------------------------
def get_ids_path(adj_matrix, start_node, goal_node):
    # First check if the start_node and goal_node are in the same connected component
    if not is_same_component(start_node, goal_node):
        # print(f"No path found from {start_node} to {goal_node} (Different components)")
        return None

    # Otherwise, perform Iterative Deepening Search
    max_depth = len(adj_matrix)  # Limit depth to the size of the graph
    for depth in range(max_depth):
        visited = set()
        result = dfs_limited(start_node, goal_node, depth, visited)
        if result:
            return result

    # No path found after all depths explored
    # print(f"No path found from {start_node} to {goal_node}")
    return None


# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]
#-----------------------------------------------------
def bfs(start, goal):
        queue = deque([start])
        visited = {start: [start]}
        while queue:
            current = queue.popleft()
            for neighbor, connected in enumerate(adj_matrix[current]):
                if connected and neighbor not in visited:
                    visited[neighbor] = visited[current] + [neighbor]
                    queue.append(neighbor)
                    if neighbor == goal:
                        return visited[neighbor]
        return None

# ----------------------------------------------------



def get_bidirectional_search_path(adj_matrix, start_node, goal_node):
    # First check if the start_node and goal_node are in the same connected component
    if not is_same_component(start_node, goal_node):
        # print(f"No path found from {start_node} to {goal_node} (Different components)")
        return None

    # Perform bidirectional BFS
    forward_path = bfs(start_node, goal_node)
    if forward_path is None:
        print(f"No path found from {start_node} to {goal_node}")
    return forward_path


# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]
# -----------------------------------------------
def dist(node1, node2):
        """ Euclidean distance between two nodes based on their (x, y) attributes. """
        try:
            if node1 in node_attributes and node2 in node_attributes:
                x1, y1 = float(node_attributes[node1]['x']), float(node_attributes[node1]['y'])
                x2, y2 = float(node_attributes[node2]['x']), float(node_attributes[node2]['y'])
                return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            else:
                # print(f"Missing attributes for node {node1} or {node2}")
                return float('inf')
        except (ValueError, KeyError) as e:
            # print(f"Error accessing coordinates for nodes {node1} or {node2}: {e}")
            return float('inf')


# -----------------------------------------------
""" h(w) = dist(u, w) + dist(w, v) where u is start, w is current, and v is goal. """

def get_astar_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(current_node):
        return dist(start_node, current_node) + dist(current_node, goal_node)
    if not is_same_component(start_node, goal_node):
        return None

    open_set = []
    heapq.heappush(open_set, (heuristic(start_node), start_node))
    came_from = {}
    g_score = {node: float('inf') for node in range(len(adj_matrix))}
    g_score[start_node] = 0
    f_score = {node: float('inf') for node in range(len(adj_matrix))}
    f_score[start_node] = heuristic(start_node)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal_node:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]  # Return the path in the correct order

        # Explore neighbors
        for neighbor, connected in enumerate(adj_matrix[current]):
            if connected:  
                tentative_g = g_score[current] + dist(current, neighbor)  # Distance-based cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None


# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]
# --------------------------------------------------
def a_star_search(open_set, came_from, g_score, heuristic_fn, goal_test, visited):
        """A* search implementation for a given direction."""
        while open_set:
            current_f, current = heapq.heappop(open_set)

            if goal_test(current):
                return current

            if current in visited:
                continue

            visited.add(current)

            for neighbor, connected in enumerate(adj_matrix[current]):
                if connected:  # There's an edge between current and neighbor
                    tentative_g = g_score[current] + dist(current, neighbor)
                    if tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic_fn(neighbor)
                        heapq.heappush(open_set, (f, neighbor))
        
        return None


# ---------------------------------------------------
def get_bidirectional_heuristic_search_path(adj_matrix, node_attributes, start_node, goal_node):
    def heuristic(start, node, goal):
        """Heuristic function h(w) = dist(u, w) + dist(w, v)."""
        return dist(start, node) + dist(node, goal)    
    # Check if the start_node and goal_node are in the same connected component
    if not is_same_component(start_node, goal_node):
        # print(f"No path found from {start_node} to {goal_node} (Different components)")
        return None

    # Initialize forward and backward heuristic searches
    open_forward = []
    open_backward = []
    heapq.heappush(open_forward, (heuristic(start_node, start_node, goal_node), start_node))
    heapq.heappush(open_backward, (heuristic(goal_node, goal_node, start_node), goal_node))
    
    came_from_forward = {start_node: None}
    came_from_backward = {goal_node: None}
    
    g_score_forward = {node: float('inf') for node in range(len(adj_matrix))}
    g_score_backward = {node: float('inf') for node in range(len(adj_matrix))}
    
    g_score_forward[start_node] = 0
    g_score_backward[goal_node] = 0

    visited_forward = set()
    visited_backward = set()
    meeting_node = None

    # Perform the bidirectional A* search
    while open_forward and open_backward:
        # Forward step
        current = a_star_search(open_forward, came_from_forward, g_score_forward, lambda n: heuristic(start_node, n, goal_node), lambda n: n in visited_backward, visited_forward)
        if current is not None:
            if current in visited_backward:
                meeting_node = current
                break

        # Backward step
        current = a_star_search(open_backward, came_from_backward, g_score_backward, lambda n: heuristic(goal_node, n, start_node), lambda n: n in visited_forward, visited_backward)
        if current is not None:
            if current in visited_forward:
                meeting_node = current
                break

    if meeting_node is None:
        print(f"No path found from {start_node} to {goal_node}")
        return None  # No path found

    # Reconstruct the path from start_node to meeting_node
    path_forward = []
    node = meeting_node
    while node is not None:
        path_forward.append(node)
        node = came_from_forward[node]

    # Reconstruct the path from meeting_node to goal_node
    path_backward = []
    node = came_from_backward[meeting_node]  # Skip meeting node to avoid duplication
    while node is not None:
        path_backward.append(node)
        node = came_from_backward[node]

    return path_forward[::-1] + path_backward  # Combine both parts of the path




# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].
def dfs(u, parent, discovery_time, low, visited, bridges, time):
        visited[u] = True
        discovery_time[u] = low[u] = time[0]
        time[0] += 1

        for v, connected in enumerate(adj_matrix[u]):
            if connected:
                if not visited[v]:
                    dfs(v, u, discovery_time, low, visited, bridges, time)
                    low[u] = min(low[u], low[v])

                    # Check if the edge u-v is a bridge
                    if low[v] > discovery_time[u]:
                        bridges.append((u, v))
                elif v != parent:
                    low[u] = min(low[u], discovery_time[v])

def bonus_problem(adj_matrix):
    
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    discovery_time = [-1] * num_nodes
    low = [-1] * num_nodes
    bridges = []
    time = [0]  # Time counter for DFS

    # Perform DFS from every node (to handle disconnected graphs)
    for i in range(num_nodes):
        if not visited[i]:
            dfs(i, -1, discovery_time, low, visited, bridges, time)

    return bridges


if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  print(f'Iterative Deepening Search Path: {get_ids_path(adj_matrix,start_node,end_node)}')
  print(f'Bidirectional Search Path: {get_bidirectional_search_path(adj_matrix,start_node,end_node)}')
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bonus Problem: {bonus_problem(adj_matrix)}')