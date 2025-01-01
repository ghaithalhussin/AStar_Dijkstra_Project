
import numpy as np
import heapq
import time

# Generate a 10x10 maze with obstacles
def generate_maze(size=10, obstacle_density=0.3):
    maze = np.zeros((size, size), dtype=int)
    num_obstacles = int(size * size * obstacle_density)
    obstacles = np.random.choice(size * size, num_obstacles, replace=False)
    for obs in obstacles:
        maze[obs // size, obs % size] = 1
    maze[0, 0] = 0  # Start point
    maze[size - 1, size - 1] = 0  # End point
    return maze

# A* algorithm implementation
def a_star(maze):
    size = len(maze)
    start, goal = (0, 0), (size - 1, size - 1)
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    explored_nodes = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        explored_nodes += 1
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        neighbors = get_neighbors(maze, current)
        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))
    return [], explored_nodes  # No path found

# Dijkstra algorithm implementation
def dijkstra(maze):
    size = len(maze)
    start, goal = (0, 0), (size - 1, size - 1)
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost = {start: 0}
    explored_nodes = 0

    while open_list:
        _, current = heapq.heappop(open_list)
        explored_nodes += 1
        if current == goal:
            return reconstruct_path(came_from, current), explored_nodes
        neighbors = get_neighbors(maze, current)
        for neighbor in neighbors:
            new_cost = cost[current] + 1
            if neighbor not in cost or new_cost < cost[neighbor]:
                came_from[neighbor] = current
                cost[neighbor] = new_cost
                heapq.heappush(open_list, (cost[neighbor], neighbor))
    return [], explored_nodes  # No path found

# Helper functions
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(maze, node):
    size = len(maze)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy
        if 0 <= x < size and 0 <= y < size and maze[x, y] == 0:
            neighbors.append((x, y))
    return neighbors

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]

# Main function
if __name__ == "__main__":
    maze = generate_maze(obstacle_density=0.3)
    print("Maze:")
    print(maze)

    start_time = time.time()
    a_star_path, a_star_nodes = a_star(maze)
    a_star_time = time.time() - start_time
    print(f"A* Path: {a_star_path}, Explored Nodes: {a_star_nodes}, Time: {a_star_time:.5f} sec")

    start_time = time.time()
    dijkstra_path, dijkstra_nodes = dijkstra(maze)
    dijkstra_time = time.time() - start_time
    print(f"Dijkstra Path: {dijkstra_path}, Explored Nodes: {dijkstra_nodes}, Time: {dijkstra_time:.5f} sec")
