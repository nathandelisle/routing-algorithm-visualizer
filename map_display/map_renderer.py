import osmnx as ox
import folium
import networkx as nx
import os
from geopy.distance import distance

def render_dynamic_map(graphml_path, output_html, start_coords, end_coords, initial_zoom=12):
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    graph = ox.load_graphml(graphml_path)

    nodes, edges = ox.graph_to_gdfs(graph)

    center = [nodes['y'].mean(), nodes['x'].mean()]
    folium_map = folium.Map(location=center, zoom_start=initial_zoom)

    def render_edges_within_bounds(folium_map, graph, bounds):
        min_lat, min_lon, max_lat, max_lon = bounds
        for _, row in edges.iterrows():
            if min_lat <= row['geometry'].coords[0][1] <= max_lat and min_lon <= row['geometry'].coords[0][0] <= max_lon:
                folium.PolyLine(locations=[(row['geometry'].coords[0][1], row['geometry'].coords[0][0]),
                                           (row['geometry'].coords[-1][1], row['geometry'].coords[-1][0])],
                                color='blue', weight=2.5).add_to(folium_map)

    initial_bounds = folium_map.get_bounds()
    render_edges_within_bounds(folium_map, graph, initial_bounds)

    start_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
    end_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

    route = nx.shortest_path(graph, start_node, end_node, weight='length')

    route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]
    folium.PolyLine(locations=route_coords, color='red', weight=2.5).add_to(folium_map)

    def update_map(event=None):
        bounds = folium_map.get_bounds()
        render_edges_within_bounds(folium_map, graph, bounds)

    folium_map.on('moveend', update_map)

    folium_map.save(output_html)
    print(f"Map rendered and saved to {output_html}")

graphml_path = 'data/map.graphml'
output_html = 'data/map_with_dynamic_route.html'
start_coords = (42.3295, -72.6740)
end_coords = (42.3495, -72.6740)

render_dynamic_map(graphml_path, output_html, start_coords, end_coords)
