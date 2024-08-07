import osmnx as ox
import folium
import json
import os
import networkx as nx
from geopy.distance import distance
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLineEdit, QPushButton, QLabel, QHBoxLayout, QMessageBox, QProgressBar, QCheckBox, QSizePolicy
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, QThread, pyqtSignal, QObject
import logging
import sys
import time
from collections import deque

logging.basicConfig(level=logging.DEBUG)

class RouteCalculationThread(QThread):
    progress = pyqtSignal(int)
    intermediate_map = pyqtSignal(str)
    final_map = pyqtSignal(str)
    finished = pyqtSignal()
    steps_calculated = pyqtSignal(int)

    def __init__(self, graphml_path, start_coords, end_coords, output_html, algorithm):
        super().__init__()
        self.graphml_path = graphml_path
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.output_html = output_html
        self.algorithm = algorithm
        self.route_finder = RouteFinder()
        self.route_finder.progress.connect(self.report_progress)
        self.route_finder.intermediate_map.connect(self.report_intermediate_map)
        self.route_finder.final_map.connect(self.report_final_map)

    def run(self):
        try:
            steps, _ = self.route_finder.calculate_route(self.graphml_path, self.start_coords, self.end_coords, self.algorithm)
            render_map_with_route(self.graphml_path, self.output_html, self.start_coords, self.end_coords)
            self.steps_calculated.emit(steps)
        except Exception as e:
            logging.error(f"Error in route calculation: {e}")
        self.finished.emit()

    def report_progress(self, value):
        self.progress.emit(value)

    def report_intermediate_map(self, path):
        self.intermediate_map.emit(path)

    def report_final_map(self, path):
        self.final_map.emit(path)

class ComparisonThread(QThread):
    progress = pyqtSignal(int)
    intermediate_map = pyqtSignal(str)
    final_map = pyqtSignal(str)
    comparison_result = pyqtSignal(dict)
    algorithm_display = pyqtSignal(str)

    def __init__(self, graphml_path, start_coords, end_coords, algorithms):
        super().__init__()
        self.graphml_path = graphml_path
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.algorithms = algorithms
        self.route_finder = RouteFinder()
        self.route_finder.progress.connect(self.report_progress)
        self.route_finder.intermediate_map.connect(self.report_intermediate_map)
        self.route_finder.final_map.connect(self.report_final_map)
        self.route_finder.comparison_result.connect(self.report_comparison_result)

    def run(self):
        results = {}
        final_routes = {}

        for algorithm in self.algorithms:
            self.algorithm_display.emit(f"Current Algorithm: {algorithm.capitalize()}")
            try:
                steps, route_coords = self.route_finder.calculate_route(self.graphml_path, self.start_coords, self.end_coords, algorithm)
                results[algorithm] = steps
                final_routes[algorithm] = route_coords

                intermediate_map_path = f'data/intermediate_map_{algorithm}.html'
                self.save_intermediate_map(route_coords, intermediate_map_path)
                self.intermediate_map.emit(intermediate_map_path)
            except Exception as e:
                results[algorithm] = str(e)
                logging.error(f"Error in {algorithm} calculation: {e}")

        final_map_path = 'data/final_map_compare_all.html'
        self.save_final_map(final_routes, final_map_path)
        self.final_map.emit(final_map_path)
        self.comparison_result.emit(results)

    def report_progress(self, value):
        self.progress.emit(value)

    def report_intermediate_map(self, path):
        self.intermediate_map.emit(path)

    def report_final_map(self, path):
        self.final_map.emit(path)

    def report_comparison_result(self, results):
        self.comparison_result.emit(results)

    def save_intermediate_map(self, route_coords, path):
        initial_location = route_coords[0]
        folium_map = folium.Map(location=initial_location, zoom_start=14)
        folium.PolyLine(route_coords, color='red', weight=2.5).add_to(folium_map)
        folium_map.save(path)

    def save_final_map(self, final_routes, path):
        initial_location = [self.start_coords[0], self.start_coords[1]]
        final_map = folium.Map(location=initial_location, zoom_start=14)
        graph = ox.load_graphml(self.graphml_path)
        for u, v, data in graph.edges(data=True):
            folium.PolyLine([(graph.nodes[u]['y'], graph.nodes[u]['x']), (graph.nodes[v]['y'], graph.nodes[v]['x'])],
                            color='blue', weight=1.5).add_to(final_map)

        for algorithm, route in final_routes.items():
            if route:
                folium.PolyLine(route, color='red', weight=2.5, tooltip=algorithm).add_to(final_map)

        final_map.save(path)

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
                logging.info(f"Step {i+1}: Node {route[i]} -> Node {route[i+1]}")
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Routing Algorithm Visualizer')
        self.setGeometry(100, 100, 800, 600)

        self.start_label = QLabel("Start (lat, lon):")
        self.end_label = QLabel("End (lat, lon):")

        self.start_input = QLineEdit()
        self.end_input = QLineEdit()

        self.route_button = QPushButton("Find Route")
        self.route_button.clicked.connect(self.find_route)

        self.dijkstra_checkbox = QCheckBox("Dijkstra's Algorithm")
        self.astar_checkbox = QCheckBox("A* Algorithm")
        self.bellman_ford_checkbox = QCheckBox("Bellman-Ford Algorithm")
        self.bfs_checkbox = QCheckBox("BFS Algorithm")
        self.dfs_checkbox = QCheckBox("DFS Algorithm")
        self.bidirectional_checkbox = QCheckBox("Bidirectional Search Algorithm")
        self.compare_checkbox = QCheckBox("Compare All")
        self.compare_checkbox.stateChanged.connect(self.compare_all_checked)

        self.dijkstra_checkbox.stateChanged.connect(self.deselect_compare_all)
        self.astar_checkbox.stateChanged.connect(self.deselect_compare_all)
        self.bellman_ford_checkbox.stateChanged.connect(self.deselect_compare_all)
        self.bfs_checkbox.stateChanged.connect(self.deselect_compare_all)
        self.dfs_checkbox.stateChanged.connect(self.deselect_compare_all)
        self.bidirectional_checkbox.stateChanged.connect(self.deselect_compare_all)

        self.dijkstra_checkbox.stateChanged.connect(self.update_route_button_state)
        self.astar_checkbox.stateChanged.connect(self.update_route_button_state)
        self.bellman_ford_checkbox.stateChanged.connect(self.update_route_button_state)
        self.bfs_checkbox.stateChanged.connect(self.update_route_button_state)
        self.dfs_checkbox.stateChanged.connect(self.update_route_button_state)
        self.bidirectional_checkbox.stateChanged.connect(self.update_route_button_state)
        self.compare_checkbox.stateChanged.connect(self.update_route_button_state)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        self.algorithm_display_label = QLabel("")
        self.algorithm_display_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.algorithm_display_label.setMaximumHeight(30)

        self.web_view = QWebEngineView()

        self.initial_map_path = os.path.join(os.getcwd(), 'data', 'map.html')
        self.load_bbox()
        self.create_initial_map()

        self.web_view.setUrl(QUrl(f'file:///{self.initial_map_path.replace("\\", "/")}'))

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.start_label)
        input_layout.addWidget(self.start_input)
        input_layout.addWidget(self.end_label)
        input_layout.addWidget(self.end_input)

        algorithm_layout = QVBoxLayout()
        algorithm_layout.addWidget(self.dijkstra_checkbox)
        algorithm_layout.addWidget(self.astar_checkbox)
        algorithm_layout.addWidget(self.bellman_ford_checkbox)
        algorithm_layout.addWidget(self.bfs_checkbox)
        algorithm_layout.addWidget(self.dfs_checkbox)
        algorithm_layout.addWidget(self.bidirectional_checkbox)
        algorithm_layout.addWidget(self.compare_checkbox)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.route_button)

        layout = QVBoxLayout()
        layout.addLayout(input_layout)
        layout.addLayout(algorithm_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.algorithm_display_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.web_view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_route_button_state()

    def load_bbox(self):
        bbox_filepath = 'data/bounding_box.json'
        with open(bbox_filepath, 'r') as f:
            self.bbox = json.load(f)

    def create_initial_map(self):
        center_lat = (self.bbox['north'] + self.bbox['south']) / 2
        center_lon = (self.bbox['east'] + self.bbox['west']) / 2

        initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        os.makedirs('data', exist_ok=True)
        initial_map.save(self.initial_map_path)
        print(f"Initial map created at {self.initial_map_path}")

    def find_route(self):
        try:
            start_coords = self.parse_coordinates(self.start_input.text())
            end_coords = self.parse_coordinates(self.end_input.text())
            self.validate_coordinates_within_bbox(start_coords)
            self.validate_coordinates_within_bbox(end_coords)
            graphml_path = 'data/map.graphml'
            output_html = 'data/map_with_route.html'

            selected_algorithms = []
            if self.dijkstra_checkbox.isChecked():
                selected_algorithms.append('dijkstra')
            if self.astar_checkbox.isChecked():
                selected_algorithms.append('astar')
            if self.bellman_ford_checkbox.isChecked():
                selected_algorithms.append('bellman_ford')
            if self.bfs_checkbox.isChecked():
                selected_algorithms.append('bfs')
            if self.dfs_checkbox.isChecked():
                selected_algorithms.append('dfs')
            if self.bidirectional_checkbox.isChecked():
                selected_algorithms.append('bidirectional')
            if self.compare_checkbox.isChecked() or not selected_algorithms:
                selected_algorithms = ['dijkstra', 'astar', 'bellman_ford', 'bfs', 'dfs', 'bidirectional']

            if not selected_algorithms:
                raise ValueError("Please select at least one algorithm.")

            self.disable_buttons()

            self.progress_bar.setVisible(True)

            if len(selected_algorithms) > 1 or self.compare_checkbox.isChecked():
                self.comparison_thread = ComparisonThread(graphml_path, start_coords, end_coords, selected_algorithms)
                self.comparison_thread.progress.connect(self.update_progress)
                self.comparison_thread.intermediate_map.connect(self.update_map)
                self.comparison_thread.final_map.connect(self.display_final_map)
                self.comparison_thread.comparison_result.connect(self.show_comparison_results)
                self.comparison_thread.algorithm_display.connect(self.update_algorithm_display)
                self.comparison_thread.start()
            else:
                self.route_thread = RouteCalculationThread(graphml_path, start_coords, end_coords, output_html, selected_algorithms[0])
                self.route_thread.progress.connect(self.update_progress)
                self.route_thread.intermediate_map.connect(self.update_map)
                self.route_thread.final_map.connect(self.display_final_map)
                self.route_thread.steps_calculated.connect(self.show_single_algorithm_steps)
                self.route_thread.finished.connect(self.on_route_calculated)
                self.route_thread.start()
        except ValueError as e:
            self.show_error_message(str(e))

    def show_single_algorithm_steps(self, steps):
        QMessageBox.information(self, "Route Steps", f"The selected algorithm took {steps} steps.")

    def compare_all_checked(self, state):
        if state == 2:
            self.deselect_all_algorithm_checkboxes()

    def deselect_compare_all(self, state):
        if state == 2:
            self.compare_checkbox.setChecked(False)

    def deselect_all_algorithm_checkboxes(self):
        self.dijkstra_checkbox.setChecked(False)
        self.astar_checkbox.setChecked(False)
        self.bellman_ford_checkbox.setChecked(False)
        self.bfs_checkbox.setChecked(False)
        self.dfs_checkbox.setChecked(False)
        self.bidirectional_checkbox.setChecked(False)

    def update_route_button_state(self):
        if (self.dijkstra_checkbox.isChecked() or self.astar_checkbox.isChecked() or self.bellman_ford_checkbox.isChecked() or
                self.bfs_checkbox.isChecked() or self.dfs_checkbox.isChecked() or self.bidirectional_checkbox.isChecked() or
                self.compare_checkbox.isChecked()):
            self.route_button.setEnabled(True)
        else:
            self.route_button.setEnabled(False)

    def parse_coordinates(self, text):
        try:
            lat, lon = map(float, text.split(','))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Coordinates are out of valid range.")
            return (lat, lon)
        except ValueError:
            raise ValueError("Invalid coordinates. Please enter valid latitude and longitude in the format: lat, lon")

    def validate_coordinates_within_bbox(self, coords):
        lat, lon = coords
        if not (self.bbox['south'] <= lat <= self.bbox['north'] and self.bbox['west'] <= lon <= self.bbox['east']):
            raise ValueError("Coordinates are outside the valid map region. Please enter coordinates within the map region.")

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_map(self, path):
        self.web_view.setUrl(QUrl(f'file:///{os.path.abspath(path).replace("\\", "/")}'))

    def display_final_map(self, path):
        self.web_view.setUrl(QUrl(f'file:///{os.path.abspath(path).replace("\\", "/")}'))
        self.progress_bar.setValue(100)

    def show_comparison_results(self, results):
        comparison_message = "Comparison of Routing Algorithms:\n"
        for algorithm, steps in results.items():
            comparison_message += f"{algorithm.capitalize()}: {steps} steps\n"

        QMessageBox.information(self, "Comparison Results", comparison_message)
        self.progress_bar.setVisible(False)
        self.algorithm_display_label.setText("")
        self.enable_buttons()

    def update_algorithm_display(self, algorithm):
        self.algorithm_display_label.setText(algorithm)

    def on_route_calculated(self):
        output_map_path = os.path.join(os.getcwd(), 'data/map_with_route.html')
        self.web_view.setUrl(QUrl(f'file:///{output_map_path.replace("\\", "/")}'))
        self.progress_bar.setValue(100)
        self.enable_buttons()

    def disable_buttons(self):
        self.route_button.setEnabled(False)
        self.dijkstra_checkbox.setEnabled(False)
        self.astar_checkbox.setEnabled(False)
        self.bellman_ford_checkbox.setEnabled(False)
        self.bfs_checkbox.setEnabled(False)
        self.dfs_checkbox.setEnabled(False)
        self.bidirectional_checkbox.setEnabled(False)
        self.compare_checkbox.setEnabled(False)

    def enable_buttons(self):
        self.route_button.setEnabled(True)
        self.dijkstra_checkbox.setEnabled(True)
        self.astar_checkbox.setEnabled(True)
        self.bellman_ford_checkbox.setEnabled(True)
        self.bfs_checkbox.setEnabled(True)
        self.dfs_checkbox.setEnabled(True)
        self.bidirectional_checkbox.setEnabled(True)
        self.compare_checkbox.setEnabled(True)

def render_map_with_route(graphml_path, output_html, start_coords, end_coords, boundary_radius_miles=50):
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    graph = ox.load_graphml(graphml_path)

    nodes, edges = ox.graph_to_gdfs(graph)

    center = [nodes['y'].mean(), nodes['x'].mean()]
    folium_map = folium.Map(location=center, zoom_start=12)

    # Calculate the extended boundary
    boundary_center = (center[0], center[1])
    boundary_north = distance(miles=boundary_radius_miles).destination(boundary_center, 0).latitude
    boundary_south = distance(miles=boundary_radius_miles).destination(boundary_center, 180).latitude
    boundary_east = distance(miles=boundary_radius_miles).destination(boundary_center, 90).longitude
    boundary_west = distance(miles=boundary_radius_miles).destination(boundary_center, 270).longitude

    # Filter edges within the extended boundary
    def within_boundary(lat, lon):
        return boundary_south <= lat <= boundary_north and boundary_west <= lon <= boundary_east

    for _, row in edges.iterrows():
        if within_boundary(row['geometry'].coords[0][1], row['geometry'].coords[0][0]) and within_boundary(row['geometry'].coords[-1][1], row['geometry'].coords[-1][0]):
            folium.PolyLine(locations=[(row['geometry'].coords[0][1], row['geometry'].coords[0][0]),
                                       (row['geometry'].coords[-1][1], row['geometry'].coords[-1][0])],
                            color='blue', weight=2.5).add_to(folium_map)

    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

    route = nx.shortest_path(graph, start_node, end_node, weight='length')

    route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
    folium.PolyLine(locations=route_coords, color='red', weight=2.5).add_to(folium_map)

    folium_map.save(output_html)
    print(f"Map rendered and saved to {output_html}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
