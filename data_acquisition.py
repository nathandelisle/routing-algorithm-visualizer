import osmnx as ox
import json
import os
import shutil

def download_map_data(place_name):
    try:
        if os.path.exists('data'):
            shutil.rmtree('data')
        os.makedirs('data', exist_ok=True)

        graph = ox.graph_from_place(place_name, network_type='all')

        filepath = 'data/map.graphml'
        ox.save_graphml(graph, filepath)

        nodes, edges = ox.graph_to_gdfs(graph)
        bbox = edges.total_bounds
        bbox_dict = {
            'north': bbox[3],
            'south': bbox[1],
            'east': bbox[2],
            'west': bbox[0]
        }

        bbox_filepath = 'data/bounding_box.json'
        with open(bbox_filepath, 'w') as f:
            json.dump(bbox_dict, f)

        print(f"Map data saved to {filepath}")
        print(f"Bounding box saved to {bbox_filepath}")
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to download map data. Please check the location name and try again.")

if __name__ == "__main__":
    place_name = input("Enter the location name (e.g., 'Milan, Lombardy, Italy'): ")
    download_map_data(place_name)
