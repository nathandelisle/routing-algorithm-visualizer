import osmnx as ox
import networkx as nx
import time
import logging
from PyQt5.QtCore import pyqtSignal, QObject
import folium
from collections import deque
import os

DETAILED_OUTPUT_MODE = True  # Set to True to enable detailed mode

logging.basicConfig(level=logging.DEBUG if DETAILED_OUTPUT_MODE else logging.INFO)
logger = logging.getLogger(__name__)

class RouteFinder(QObject):
    progress = pyqtSignal(int)
    intermediate_map = pyqtSignal(str)
    final_map = pyqtSignal(str)
    comparison_result = pyqtSignal(dict)

    def __init__(self):
        super().__init__()

    def calculate_route(self, graphml_path, start_coords, end_coords, algorithm):
        graph = ox.load_graphml(graphml_path)
        start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
        end_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

        self.progress.emit(0)

        try:
            if algorithm == 'dijkstra':
                route = nx.shortest_path(graph, start_node, end_node, weight='length')
            elif algorithm == 'astar':
                route = nx.astar_path(graph, start_node, end_node, weight='length')
            elif algorithm == 'bellman_ford':
                route = nx.bellman_ford_path(graph, start_node, end_node, weight='length')
            elif algorithm == 'bfs':
                route = self.bfs(graph, start_node, end_node)
            elif algorithm == 'dfs':
                route = self.dfs(graph, start_node, end_node)
            elif algorithm == 'bidirectional':
                route = self.bidirectional_search(graph, start_node, end_node)
            else:
                raise ValueError("Unknown algorithm selected")

            total_nodes = len(route)
            route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]

            for i in range(len(route_coords) - 1):
                if DETAILED_OUTPUT_MODE:
                    logger.info(f"Step {i+1}: Node {route[i]} -> Node {route[i+1]}")

                time.sleep(0.1)
                progress_percentage = int((i + 1) / total_nodes * 100)
                self.progress.emit(progress_percentage)

                os.makedirs('data', exist_ok=True)
                intermediate_map = folium.Map(location=route_coords[i+1], zoom_start=15)
                folium.PolyLine(route_coords[:i+2], color='red', weight=2.5).add_to(intermediate_map)
                intermediate_map_path = f'data/intermediate_map_{algorithm}_step_{i+1}.html'
                intermediate_map.save(intermediate_map_path)
                self.intermediate_map.emit(intermediate_map_path)

            self.progress.emit(100)
            return total_nodes - 1, route_coords
        except Exception as e:
            raise ValueError(f"An error occurred during route calculation: {str(e)}")

    def bfs(self, graph, start, goal):
        queue = deque([start])
        visited = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited[neighbor] = current
        return self.reconstruct_path(visited, start, goal)

    def dfs(self, graph, start, goal):
        stack = [start]
        visited = {start: None}
        while stack:
            current = stack.pop()
            if current == goal:
                break
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
                    visited[neighbor] = current
        return self.reconstruct_path(visited, start, goal)

    def bidirectional_search(self, graph, start, goal):
        if start == goal:
            return [start]

        forward_queue = deque([start])
        backward_queue = deque([goal])
        forward_visited = {start: None}
        backward_visited = {goal: None}

        while forward_queue and backward_queue:
            if forward_queue:
                current_forward = forward_queue.popleft()
                for neighbor in graph.neighbors(current_forward):
                    if neighbor not in forward_visited:
                        forward_queue.append(neighbor)
                        forward_visited[neighbor] = current_forward
                        if neighbor in backward_visited:
                            return self.reconstruct_bidirectional_path(forward_visited, backward_visited, neighbor, start, goal)

            if backward_queue:
                current_backward = backward_queue.popleft()
                for neighbor in graph.neighbors(current_backward):
                    if neighbor not in backward_visited:
                        backward_queue.append(neighbor)
                        backward_visited[neighbor] = current_backward
                        if neighbor in forward_visited:
                            return self.reconstruct_bidirectional_path(forward_visited, backward_visited, neighbor, start, goal)

        raise ValueError("No path found between the given nodes")

    def reconstruct_path(self, visited, start, goal):
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = visited[current]
        path.reverse()
        return path

    def reconstruct_bidirectional_path(self, forward_visited, backward_visited, meeting_point, start, goal):
        path = []

        current = meeting_point
        while current is not None:
            path.append(current)
            current = forward_visited[current]
        path.reverse()

        current = backward_visited[meeting_point]
        while current is not None:
            path.append(current)
            current = backward_visited[current]

        return path
