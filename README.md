Routing Algorithm Visualizer


Overview:
The Routing Algorithm Visualizer is a Python application for visualization of graph traversal algorithms. It finds routes between two locations, using OSMnx for map data, NetworkX for graph operations, and Folium for map visualization. 

The application supports Dijkstra, A*, Bellman-Ford, BFS, DFS, and Bidirectional Search, and allows for comparison of these algorithms based on the number of steps taken to find a route.


Installation:
Clone repo: git clone https://github.com/your-username/routing-algorithm-visualizer.git
Install dependencies: pip install -r requirements.txt


Usage:
To download map data for a region, first run: python data_acquisition.py
You will be prompted to enter a location. Then, you can run python main.py to start the application.

When Detailed Output Mode is enabled, the application will log detailed information to the console. To disable, set DETAILED_OUTPUT_MODE in route_finder.py to False.


I hope you enjoy and find this cool. :)
