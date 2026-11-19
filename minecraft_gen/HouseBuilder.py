from gdpc import Editor, Block
import numpy as np
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter

from minecraft_gen.WorldMap import WorldMap
from collections import Counter

UNOCCUPIED = 0
HOUSE = 2
BORDER = 1
DOOR = 3

VALID_COST_THRESHOLD_PERC = 75 # Tune this value
SMOOTHING_SIGMA = 5
INFINITY = 1e6

PILLAR_BLOCK_CHOICES = ["quartz_pillar"]
FOUNDATION_BLOCK_CHOICES = ["stone", "andesite", "cobblestone", "sandstone"]

class HouseBuilder:
    def __init__(self, world_map: WorldMap):
        self.DOOR = DOOR
        self.HOUSE = HOUSE
        
        self.world_map = world_map
        self.map_data = None
        self.building_spots = None

    # TODO: Implement rotate_building method  
    def __rotate_building(self, building: dict, orientation: str) -> dict:
        """
        Rotate a building clockwise in 90-degree increments.
        
        Args:
            building (dict): A dictionary containing building information, including size and door position.
            orientation (str): The desired orientation ("N", "E", "S", "W").
        
        Returns:
            dict: A new dictionary with the rotated building information, including updated size and door position.
        """

        size_x, size_y, size_z = building["size"]
        door_x, door_z = building["door_pos"]
        orientation_map = {"N": 0, "E": 1, "S": 2, "W": 3}
        
        current_orientation = orientation_map[self.__get_door_orientation(building["size"], building["door_pos"])]

        turn = (orientation_map[orientation] - current_orientation) % 4

        if turn == 0:
            new_size = (size_x, size_y, size_z)
            new_door_pos = (door_x, door_z)

        elif turn == 1:  # 90°
            new_size = (size_z, size_y, size_x)
            new_door_pos = (
                door_z,
                size_x - 1 - door_x,
            )

        elif turn == 2:  # 180°
            new_size = (size_x, size_y, size_z)
            new_door_pos = (
                size_x - 1 - door_x,
                size_z - 1 - door_z,
            )

        else:  # 270°
            new_size = (size_z, size_y, size_x)
            new_door_pos = (
                size_z - 1 - door_z,
                door_x,
            )

        rotated = building.copy()
        rotated["size"] = new_size
        rotated["door_pos"] = new_door_pos
        rotated["rotation"] = turn

        return rotated

    def __get_door_orientation(self, size, door_pos):
        size_x, _, size_z = size
        door_x, door_z = door_pos

        if door_z == 0:
            return "S"
        if door_x == size_x - 1:
            return "E"
        if door_z == size_z - 1:
            return "N"
        if door_x == 0:
            return "W"

    def __plot_placement(self, placement_map, slope_scores):
        """Plot the building placement map."""

        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(placement_map.T, cmap='coolwarm')
        ax.invert_yaxis()
        # ax.set_title("Building Placement", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/placement_map.pdf")
        plt.show()

        # plt.figure(figsize=(8, 8))
        # ax = sns.heatmap(placement_map.T, cmap="coolwarm", cbar=False, square=True,
        #                 xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()  # Invert the Y-axis
        # plt.title("Building Placement")
        # # plt.axis('off')
        # ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        # ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        # plt.show()

        slope_scores[slope_scores == INFINITY] = 0
        
        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(slope_scores.T, cmap='coolwarm')
        ax.invert_yaxis()
        # ax.set_title("Smoothed Slope Scores", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/slope_scores2.pdf")
        plt.show()

        # plt.figure(figsize=(8, 8))
        # ax = sns.heatmap(slope_scores.T, cmap="coolwarm", cbar=False, square=True,
        #                 xticklabels=False, yticklabels=False)
        # ax.invert_yaxis()  # Invert the Y-axis
        # plt.title("Slope Scores")
        # # plt.axis('off')
        # ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        # ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        # plt.show()

    def __plot_maps(self, heightmap, slope_scores, water_masks, tree_masks):
        """Plot the heightmap, slope scores, water masks, and tree masks."""

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(heightmap.T, cmap='coolwarm')
        fig.colorbar(cax, ax=ax, label='Value')
        ax.invert_yaxis()
        # ax.set_title("Height Map", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/heightmap.pdf")
        plt.show()
        
        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(slope_scores.T, cmap='coolwarm')
        fig.colorbar(cax, ax=ax, label='Value')
        ax.invert_yaxis()
        # ax.set_title("Smoothed Slope Scores", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/slope_scores.pdf")
        plt.show()

        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(water_masks.T, cmap="gray", vmin=0, vmax=1, origin="lower")
        # ax.set_title("Water/Lava Positions", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/water_mask.pdf")
        plt.show()

        fig, ax = plt.subplots(layout='constrained')
        cax = ax.imshow(tree_masks.T, cmap="gray", vmin=0, vmax=1, origin="lower")
        # ax.set_title("Tree Positions", size=18)
        ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
        ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
        plt.savefig("./minecraft_gen/tree_mask.pdf")
        plt.show()

        
    # --------------------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------------------
    def get_placement_map(self, building_types: list[dict], plot=True, save=False) -> tuple[np.ndarray, list[dict]]:
        """
        building_types: [
            {"name": <str>, 
             "size": (x, y, z), 
             "max_num": <int>, 
             "border": <float>, 
             "door_pos": (x, z), 
             "y_offset": <int>},
             "blocks": <list of block entries>}
        ]
        "center" are built first (if exists), then the rest are built around it. 
        """
        placement_map = np.full((self.world_map.build_area_size[0], self.world_map.build_area_size[2]), fill_value=UNOCCUPIED, dtype=int)
        building_spots = [] # list of dicts {name: str, start_position: (x, z), size: (x, z), orientation: Literal["N", "E", "S", "W"], y_offset: int}

        # sort building types by size and put center building first (if exists)
        building_types = sorted(building_types, key=lambda b: ("center" not in b["name"], b["size"][0] * b["size"][2]))
        smoothed_slope_scores = gaussian_filter(self.world_map.slope_scores.astype(np.float64, copy=True), sigma=SMOOTHING_SIGMA)
        modified_slope_scores = smoothed_slope_scores.copy()
        threshold = np.percentile(modified_slope_scores, VALID_COST_THRESHOLD_PERC)


        map_center_x = modified_slope_scores.shape[0] // 2
        map_center_z = modified_slope_scores.shape[1] // 2

        for building_type in building_types:
            building_name = building_type["name"]
            building_size = building_type["size"]
            max_num_buildings = building_type["max_num"]
            border = building_type["border"]
            door_pos = building_type["door_pos"]
            y_offset = building_type["y_offset"]

            # Create a kernel for the building size
            buffer = 6
            kernel = np.ones((building_size[0] + 2*border + buffer, building_size[2] + 2*border + buffer), dtype=float)
            placed_buildings = 0
            while placed_buildings < max_num_buildings:
                # Compute the cost map by correlating the kernel with the slope scores
                cost_map = correlate2d(modified_slope_scores, kernel, mode='valid', boundary='fill', fillvalue=INFINITY)
                # Find the best placement position (lowest cost)
                best_position = np.unravel_index(np.argmin(cost_map), cost_map.shape)
                # Check if the best position is valid
                start_x = best_position[0] + border + buffer // 2
                start_z = best_position[1] + border + buffer // 2
                end_x = start_x + building_size[0]
                end_z = start_z + building_size[2]
                cost_threshold = threshold * (building_size[0] * building_size[2])
                if end_x < modified_slope_scores.shape[0] and \
                   end_z < modified_slope_scores.shape[1] and \
                    cost_map[best_position] < cost_threshold and \
                    np.all(placement_map[start_x:end_x, start_z:end_z] == 0) and \
                    np.all(self.world_map.water_mask[start_x:end_x, start_z:end_z] == 0):
                    
                    # Get orientation to the map center
                    center_x = start_x + building_size[0] // 2
                    center_z = start_z + building_size[2] // 2
                    direction_x = map_center_x - center_x
                    direction_z = map_center_z - center_z

                    # Face along the axis with the greatest distance.
                    if abs(direction_x) > abs(direction_z):
                        orientation = "E" if direction_x > 0 else "W"
                    else:
                        orientation = "N" if direction_z > 0 else "S"

                    # Rotate the building and door position based on the orientation
                    building_type_rotated = self.__rotate_building(building_type, orientation)
                    # Retrieve the rotated building information
                    building_size = building_type_rotated["size"]
                    door_pos = building_type_rotated["door_pos"]
                    end_x = start_x + building_size[0]
                    end_z = start_z + building_size[2]

                    # Mark the border area in the placement map
                    border_start_x = max(0, start_x - border)
                    border_end_x = min(modified_slope_scores.shape[0], end_x + border)
                    border_start_z = max(0, start_z - border)
                    border_end_z = min(modified_slope_scores.shape[1], end_z + border)
                    placement_map[border_start_x:border_end_x, border_start_z:border_end_z] = BORDER

                    # Mark the house area in the placement map
                    placement_map[start_x:end_x, start_z:end_z] = HOUSE

                    # Add door position
                    door_x = start_x + door_pos[0]
                    door_z = start_z + door_pos[1]
                    placement_map[door_x, door_z] = DOOR
                    
                    orientation = self.__get_door_orientation(building_size, door_pos)

                    # Enforce penalty for the border in the modified slope scores
                    modified_slope_scores[border_start_x:border_end_x, border_start_z:border_end_z] = INFINITY

                    placed_buildings += 1

                    building_spots.append({
                        "name": building_name,
                        "top_left": (start_x, start_z),
                        "size": (building_size[0], building_size[2]),
                        "border": border,
                        "orientation": orientation,
                        "y_offset": y_offset
                    })
                    
                else:
                    # No valid position found, break the loop
                    break

        if plot:
            self.__plot_maps(self.world_map.heightmap, smoothed_slope_scores, self.world_map.water_mask, self.world_map.tree_mask)
            self.__plot_placement(placement_map, modified_slope_scores)
        if save:
            np.savetxt("./minecraft_gen/placement_map.txt", placement_map, fmt="%d")

        self.map_data = placement_map
        self.building_spots = building_spots

        return placement_map, building_spots

    # Placing buildings in Minecraft
    def clear_and_set_foundation(self, building_spot_dict, foundation_block_id):
        """
        Clear the area above the building spot and set the foundation blocks.
        """
        original_heightmap = self.world_map.heightmap.copy()
        modified_heightmap = original_heightmap.copy()

        start_x, start_z = building_spot_dict["top_left"]
        size_x, size_z = building_spot_dict["size"]

        end_x = start_x + size_x
        end_z = start_z + size_z

        platform_y = int(np.rint(np.percentile(original_heightmap[start_x:end_x, start_z:end_z], 90)))

        top_block = FOUNDATION_BLOCK_CHOICES[foundation_block_id]
        core_block = "minecraft:grass_block"

        for x in range(start_x, end_x):
            for z in range(start_z, end_z):
                terrain_y = int(original_heightmap[x, z])

                # Clear space above
                for y in range(platform_y, platform_y + 30):
                    self.world_map.place_block((x, y, z), Block("minecraft:air"))

                # Place the top foundation
                self.world_map.place_block((x, platform_y - 1, z), Block(top_block))
                for y in reversed(range(terrain_y - 1, platform_y - 1)):
                    self.world_map.place_block((x, y, z), Block(core_block))

                # The building begins above the foundation surface.
                modified_heightmap[x, z] = platform_y

        self.world_map.heightmap = modified_heightmap
        return modified_heightmap


    def place_building(self, building_spot_dict, building_blocks):
        """
        Place a building in the Minecraft world based on the provided building spot and block data.
        """
        start_x, start_z = building_spot_dict["top_left"]
        size_x, size_z = building_spot_dict["size"]
        orientation = building_spot_dict["orientation"]
        y_offset = building_spot_dict["y_offset"]

        # Get the platform height from the heightmap (assuming the heightmap is updated after foundation placement)
        platform_y = int(self.world_map.heightmap[start_x, start_z])

        # Place the building blocks based on the orientation
        for block_entry in building_blocks:
            rel_x, rel_y, rel_z, block_id = block_entry
            if orientation == "N":
                world_x = start_x + rel_x
                world_y = platform_y + rel_y + y_offset
                world_z = start_z + rel_z
            elif orientation == "E":
                world_x = start_x + (size_z - 1 - rel_z)
                world_y = platform_y + rel_y + y_offset
                world_z = start_z + rel_x
            elif orientation == "S":
                world_x = start_x + (size_x - 1 - rel_x)
                world_y = platform_y + rel_y + y_offset
                world_z = start_z + (size_z - 1 - rel_z)
            elif orientation == "W":
                world_x = start_x + rel_z
                world_y = platform_y + rel_y + y_offset
                world_z = start_z + (size_x - 1 - rel_x)

            self.world_map.place_block((world_x, world_y, world_z), Block(block_id))