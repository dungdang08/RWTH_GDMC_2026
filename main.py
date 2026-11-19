#  /setbuildarea ~0 ~0 ~0 ~200 ~200 ~200
from gdpc import Editor, Block
import numpy as np
import random

from minecraft_gen.WorldMap import WorldMap
from minecraft_gen.HouseBuilder import HouseBuilder
from minecraft_gen.PathBuilder import PathBuilder
from minecraft_gen.HouseGenerator import HouseGenerator


if __name__ == "__main__":

    # # Load and process world map
    ED = Editor(buffering=True, caching=True)
    world_map = WorldMap(ED)

    try:
        world_map.load_world_slice()
        world_map.compute_slope_map(save=True)
        # world_map.load_world_map(
        #     heightmap_file="./minecraft_gen/heightmap.txt",
        #     slope_scores_file="./minecraft_gen/slope_scores.txt",
        #     tree_mask_file="./minecraft_gen/tree_mask.txt",
        #     water_mask_file="./minecraft_gen/water_mask.txt"
        # )
    except Exception as e:
        print(f"Error during world map processing: {e}")
        exit(1)

    # Generate buildings from templates
    template_path_dict = {
        "./minecraft_gen/templates/fhouse1.schem": 3,
        "./minecraft_gen/templates/fhouse2.schem": 3,
        "./minecraft_gen/templates/fhouse3.schem": 3,
        "./minecraft_gen/templates/fhouse4.schem": 3,
        "./minecraft_gen/templates/fhouse5.schem": 3,
        "./minecraft_gen/templates/fhouse6.schem": 3,
        "./minecraft_gen/templates/fhouse7.schem": 3
    }
    house_generator = HouseGenerator()
    try:
        building_types = house_generator.generate_buildings_from_templates(template_path_dict)
    except Exception as e:
        print(f"Error during building generation: {e}")
        exit(1)

    # Load building types (temporary)
    # size_w = size_d = size_h = 24
    # door_pos = (size_w // 2, 0)
    # BUILDING_TYPES = [
    #     {'name': f'fhouse{i}', 'size': (size_w, size_h, size_d), 'max_num': 3, 'border': 5, 'door_pos': door_pos, 'y_offset': 3}
    #     for i in range(1, 8)
    # ]

    # Find building placement spots
    house_builder = HouseBuilder(world_map)
    try:
        building_placement_map, building_spots = house_builder.get_placement_map(building_types, plot=True, save=True)
    except Exception as e:
        print(f"Error during building placement: {e}")
        exit(1)
    
    path_builder = PathBuilder(world_map, house_builder)
    try:
        final_paths = path_builder.make_paths_mst(alter_iterations=50, plot=True, save=True)
    except Exception as e:
        print(f"Error during path generation: {e}")
        exit(1)
    
    # Smooth the terrain around buildings and paths
    try:
        for building_spot in building_spots:
            print(f"Clearing and setting foundation for building at {building_spot['top_left']}")
            house_builder.clear_and_set_foundation(building_spot, foundation_block_id=3)
    except Exception as e:
        print(f"Error during terrain smoothing: {e}")
        exit(1)

    # Build the paths
    try:
        path_builder.build_paths(final_paths, path_block_id=3, path_water_block_id=2)
    except Exception as e:
        print(f"Error during path building: {e}")
        exit(1)

    # ============================================ TEST ==============================================

    # ED = Editor(buffering=True, caching=True)
    # world_map = WorldMap(ED)
    # world_map.load_world_slice()    
    # # Place a test block to verify the world slice is loaded correctly
    # for x in range(0, 5):
    #     for z in range(0, 5):
    #         y = int(world_map.heightmap[x, z]) - 1  # Place block above the terrain
    #         world_map.place_block((x, y, z), Block("redstone_block"))
    