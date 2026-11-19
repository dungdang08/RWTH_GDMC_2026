import heapq
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from gdpc import Block

from minecraft_gen.WorldMap import WorldMap
from minecraft_gen.HouseBuilder import HouseBuilder

UNOCCUPIED = 0
HOUSE = 2
BORDER = 1
DOOR = 3
PATH = 4

PATH_SMOOTHING_SIGMA = 2

HOUSE_AVOID_MULT = 5
SLOPE_AVOID_MULT = 0.5
WATER_AVOID_MULT = 3

WATER_ASTAR_COST = 10
BORDER_ASTAR_COST = 3

PATH_BLOCK_CHOICES = ["minecraft:sandstone", "minecraft:gravel", "minecraft:andesite", "minecraft:cobblestone"]
SLAB_BLOCK_CHOICES = ["minecraft:sandstone_slab", "minecraft:gravel_slab", "minecraft:andesite_slab", "minecraft:cobblestone_slab"]
PATH_WATER_BLOCK_CHOICES = ["minecraft:oak_log", "minecraft:birch_log", "minecraft:jungle_log", "minecraft:spruce_log", "minecraft:dark_oak_log"]


class PathBuilder:
    def __init__(self, world_map: WorldMap, house_builder: HouseBuilder):
        self.world_map = world_map
        self.house_builder = house_builder
        self.editor = world_map.editor

    # -------------------------------------------------------------------------------------
    # Source: https://github.com/IsaacBraamGit/gdcm2025/blob/main/Water_sim.py
    # -------------------------------------------------------------------------------------
    def __heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def __a_star(self, start, goal):
        if self.house_builder.map_data is None:
            raise ValueError("map_data is not set. Please run get_placement_map() first.")
        
        watermask = self.world_map.water_mask.copy()
        map_data = self.house_builder.map_data.copy()

        rows, cols = map_data.shape
        open_set = []
        heapq.heappush(open_set, (self.__heuristic(start, goal), 0, start, [start]))

        visited = set()

        while open_set:
            est_total, curr_cost, current, path = heapq.heappop(open_set)

            if current == goal:
                return path, curr_cost

            if current in visited:
                continue
            visited.add(current)

            r, c = current
            neighbors = [
                (r + dr, c + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if not (dr == 0 and dc == 0)
            ]

            for nr, nc in neighbors:
                if 0 <= nr < rows and 0 <= nc < cols:
                    if map_data[nr, nc] == HOUSE:  # Avoid buildings
                        continue

                    step_cost = 1.0
                    # Allow path to go through border but with a penalty
                    if map_data[nr, nc] == BORDER:
                        step_cost += BORDER_ASTAR_COST
                    # Allow path to go through water but with a penalty
                    if watermask is not None and watermask[nr, nc]:
                        step_cost += WATER_ASTAR_COST

                    next_cost = curr_cost + step_cost
                    
                    heapq.heappush(open_set, (
                        next_cost + self.__heuristic((nr, nc), goal),
                        next_cost,
                        (nr, nc),
                        path + [(nr, nc)]
                    ))

        return [], float('inf')

    def __new_spot(self, path, indx, path_array):
        rows, cols = path_array.shape
        row, col = path[indx]

        if row < 0 or row >= rows or col < 0 or col >= cols:
            return None

        neighbors = [
            (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
            (row, col - 1), (row, col + 1),
            (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)
        ]

        new_neighbors = []
        for n in neighbors:
            if 0 <= n[0] < rows and 0 <= n[1] < cols:
                if (
                    abs(n[0] - path[indx + 1][0]) + abs(n[1] - path[indx + 1][1]) > 0 and
                    abs(n[0] - path[indx - 1][0]) + abs(n[1] - path[indx - 1][1]) > 0 and
                    self.house_builder.map_data[n[0], n[1]] != HOUSE
                ):
                    new_neighbors.append(n)

        max_value = float("-inf")
        max_index = None

        for r, c in new_neighbors:
            value = path_array[r, c]
            if value > max_value:
                max_value = value
                max_index = (r, c)

        return max_index if max_index else path[indx]


    def __alter_path(self, path, path_array):
        if len(path) < 3:
            return path

        start = path[0]
        end = path[-1]

        for i in range(1, len(path) - 2):
            new_p = self.__new_spot(path, i, path_array)
            if new_p:
                path[i] = new_p

        new_path = []
        for i in range(len(path) - 1):
            new_path.append(path[i])
            if (abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1])) > 2:
                new_path.append((
                    (path[i][0] + path[i + 1][0]) // 2,
                    (path[i][1] + path[i + 1][1]) // 2,
                ))

        # Ensure start and end are explicitly added
        new_path = [start] + new_path[1:-1] + [end]

        return list(dict.fromkeys(new_path))


    def __fix_path_final(self, path):
        new_path = []
        for i in range(len(path) - 1):
            new_path.append(path[i])
            if (abs(path[i][0] - path[i + 1][0]) + abs(path[i][1] - path[i + 1][1])) > 1:
                new_path.append((
                    (path[i][0] + path[i + 1][0]) // 2,
                    (path[i][1] + path[i + 1][1]) // 2,
                ))
        return new_path


    def __get_neighbour_paths(self, paths, map_avoid, water_avoid):
        slope = gaussian_filter(self.world_map.slope_scores.astype(float, copy=True), sigma=PATH_SMOOTHING_SIGMA)
        # slope = self.world_map.slope_scores.astype(float, copy=True)
        rows, cols = map_avoid.shape
        path_array = np.zeros((rows, cols))

        for path in paths:
            for r, c in path:
                if 0 <= r < rows and 0 <= c < cols:
                    path_array[r, c] = 5
        
        path_array = gaussian_filter(path_array.astype(float), sigma=PATH_SMOOTHING_SIGMA, mode='constant')
        path_array -= map_avoid * HOUSE_AVOID_MULT
        path_array -= slope[:rows, :cols] * SLOPE_AVOID_MULT
        path_array -= water_avoid * WATER_AVOID_MULT

        return path_array
    # -------------------------------------------------------------------------------------
    #  End of source used from.
    # -------------------------------------------------------------------------------------

    def __min_max_spanning_tree_with_a_star_cost(self):
        """
        Find path routes between doors using a minimum and maximum spanning tree with A* cost as the edge weight.

        Returns:
            list: A list of paths connecting the doors in the minimum and maximum spanning tree.
        """

        # Find the doors of the buildings
        if self.house_builder.map_data is None:
            raise ValueError("map_data is not set. Please run get_placement_map() first.")
        
        map_data = self.house_builder.map_data.copy()
        doors = []
        for x, y in np.ndindex(map_data.shape):
            if map_data[x, y] == self.house_builder.DOOR:
                doors.append((x, y))

        edges = []
        for i in range(len(doors)):
            for j in range(i + 1, len(doors)):
                path, cost = self.__a_star(doors[i], doors[j])
                if path and cost < float("inf"):
                    edges.append((cost, i, j, path))
        # Sort edges by cost
        desc_edges = sorted(edges, key=lambda x: x[0], reverse=True)
        asc_edges = sorted(edges, key=lambda x: x[0])
        
        # Kruskal's algorithm to build MST
        def find(parent, i):
            if parent[i] == i:
                return parent, i
            else:
                parent, root = find(parent, parent[i])
                parent[i] = root  # Path compression
                return parent, root
        
        parent = {door: door for door in doors}
        mst_paths = []
        for cost, start, end, path in desc_edges:
            parent, root_start = find(parent, doors[start])
            parent, root_end = find(parent, doors[end])
            if root_start != root_end:
                parent[root_end] = root_start
                mst_paths.append(path)

        parent = {door: door for door in doors}
        for cost, start, end, path in asc_edges:
            parent, root_start = find(parent, doors[start])
            parent, root_end = find(parent, doors[end])
            if root_start != root_end:
                parent[root_end] = root_start
                if path not in mst_paths:
                    mst_paths.append(path)
            
        return mst_paths


    def make_paths_mst(self, alter_iterations=10, plot=False, save=False):
        """
        Generate paths between doors using a minimum and maximum spanning tree with A* cost as the edge weight.

        Args:
            alter_iterations (int): Number of iterations to alter the paths for optimization.
            plot (bool): Whether to plot the final paths on the map.
            save (bool): Whether to save the final paths to a text file.
        """
        if self.house_builder.map_data is None:
            raise ValueError("map_data is not set. Please run get_placement_map() first.")
        
        map_data = self.house_builder.map_data.copy().astype(float)
        rows, cols = map_data.shape
        water_mask = self.world_map.water_mask[:rows, :cols].astype(float)
        
        # Get MST paths directly
        paths = self.__min_max_spanning_tree_with_a_star_cost()

        if not paths:
            raise ValueError("Could not generate any valid paths between doors.")

        for _ in range(alter_iterations):
            for i in range(len(paths)):
                path_array = self.__get_neighbour_paths(paths, map_data, water_mask)
                paths[i] = self.__alter_path(paths[i], path_array)

        mapcopy = map_data.copy()
        final_paths = np.zeros_like(map_data, dtype=int)

        for p in paths:
            p = self.__fix_path_final(p)
            for r, c in p:
                if 0 <= r < rows and 0 <= c < cols:
                    mapcopy[r, c] = PATH
                    final_paths[r, c] = 1
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(layout='constrained')
            cax = ax.imshow(mapcopy.T, cmap='hot', interpolation='nearest')
            # fig.colorbar(cax, ax=ax, label='Value')
            ax.invert_yaxis()
            # ax.set_title("Final Paths on Map")
            ax.set_xlabel("Minecraft X: East → (positive) | West ← (negative)")
            ax.set_ylabel("Minecraft Z: South → (positive) | North ← (negative)")
            if save:
                plt.savefig("./minecraft_gen/final_paths_min_max.pdf")
            plt.show()
        if save:
            np.savetxt("./minecraft_gen/final_paths.txt", final_paths, fmt="%d")
        return final_paths
    
    def build_paths(self, final_paths, path_block_id=0, path_water_block_id=0):
        """
        Build paths in the Minecraft world based on the provided final paths array.
        
        Args:
            final_paths (np.ndarray): A 2D array indicating where paths should be built (1 for path, 0 for no path).
            path_block_id (int): Index to select the type of block to use for the path.
            path_water_block_id (int): Index to select the type of block to use for paths over water.

        Returns:
            set: A set of (x, z) tuples representing the columns where paths were built.
        """
        # Build a 3x3 path
        if self.house_builder.map_data is None:
            raise ValueError("map_data is not set. Please run get_placement_map() first.")
        
        height_map = self.world_map.heightmap
        dxz_range = (0, -1, 1)

        path_block = PATH_BLOCK_CHOICES[path_block_id%len(PATH_BLOCK_CHOICES)]
        path_water_block = PATH_WATER_BLOCK_CHOICES[path_water_block_id%len(PATH_WATER_BLOCK_CHOICES)]
        slab_block = SLAB_BLOCK_CHOICES[path_block_id%len(SLAB_BLOCK_CHOICES)] + "[type=bottom]"

        path_columns = set()  # All (x, z) columns used by path (saved for path decoration later)
        shape_x, shape_z = final_paths.shape

        sleep_counter = 0
        for x in range(shape_x):
            for z in range(shape_z):
                current_height = height_map[x, z] - 1
                if final_paths[x, z] != 1:
                    continue

                for dx in dxz_range:
                    for dz in dxz_range:
                        nx, nz = x + dx, z + dz
                        if not (0 <= nx < shape_x and 0 <= nz < shape_z):
                            continue
                        ny = height_map[nx, nz] - 1
                        # Avoid trees in the path area
                        if dx != 0 and dz != 0 and self.world_map.tree_mask[nx, nz]:
                            continue
                        # Clear space above
                        for dy in (1, 2, 3, 4, 5, 6):
                            self.world_map.place_block((nx, current_height + dy, nz), Block("air"))
                        # Place path block
                        if self.world_map.water_mask[nx, nz]:
                            self.world_map.place_block((nx, ny, nz), Block(path_water_block))
                        else:
                            for dy in range(min(current_height, ny), max(current_height, ny) + 1):
                                self.world_map.place_block((nx, dy, nz), Block(path_block))
                        path_columns.add((nx, nz))
                        sleep_counter += 1
                if sleep_counter > 100:
                    sleep_counter = 0
                    time.sleep(0.5)  # Sleep to avoid overloading the editor
        
        return path_columns       