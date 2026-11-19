#  /setbuildarea ~0 ~0 ~0 ~250 ~100 ~250
from gdpc import Editor, Block, geometry, Box, WorldSlice
from gdpc.vector_tools import Box

import numpy as np

X = 0
Y = 1
Z = 2

HEIGHTMAP_TYPE = "MOTION_BLOCKING_NO_LEAVES"
HEIGHT_SMOOTHING_SIGMA = 3
SLOPE_SMOOTHING_SIGMA = 2

TREE_LIST = ["log", "wood", "mushroom", "mushroom", "fungus"]
WATER_LIST = ["water", "lava"]

WATER_PENALTY_FACTOR = 1
TREE_PENALTY = 10 

class WorldMap:
    def __init__(self, editor: Editor):
        self.editor                                     = editor
        self.build_area : Box                           = Box((0, 0, 0), (250, 100, 250))
        self.worldslice : WorldSlice                    = None
        self.build_area_size : tuple[int, int, int]     = self.build_area.size   # (sizeX, sizeY, sizeZ)
        self.build_area_offset : tuple[int, int, int]   = self.build_area.offset # (startX, startY, startZ)

        self.heightmap : np.ndarray                     = np.zeros((self.build_area.size[0], self.build_area.size[2]), dtype=int)
        self.slope_scores : np.ndarray                  = np.zeros((self.build_area.size[0], self.build_area.size[2]), dtype=int)
        
        self.tree_mask : np.ndarray                     = np.zeros((self.build_area_size[X], self.build_area_size[Z]), dtype=bool)
        self.water_mask : np.ndarray                    = np.zeros((self.build_area_size[X], self.build_area_size[Z]), dtype=bool)

    # -----------------------------------------------------------
    # Private methods
    # -----------------------------------------------------------
            
    def __is_water(self, x, y, z):
        block_type = self.worldslice.getBlock((x, y - 1, z))
        return any(water in block_type.id for water in WATER_LIST)

    def __is_tree(self, x, y, z):
        block_type = self.worldslice.getBlock((x, y - 1, z))
        return any(tree in block_type.id for tree in TREE_LIST)

    def __compute_tree_and_water_masks(self):
        for x in range(self.heightmap.shape[0]):
            for z in range(self.heightmap.shape[1]):
                current_height = self.heightmap[(x, z)]
                if self.__is_water(x, current_height, z):
                    self.water_mask[x, z] = True
                elif self.__is_tree(x, current_height, z):
                    self.tree_mask[x, z] = True
    
    def __squash_trees_in_heightmap(self):
        for x in range(self.heightmap.shape[0]):
            for z in range(self.heightmap.shape[1]):
                if self.tree_mask[x, z]:
                    # Get the 3x3 neighborhood around the tree block
                    neighborhood = self.heightmap[max(0, x-1):min(self.heightmap.shape[0], x+2),
                                                  max(0, z-1):min(self.heightmap.shape[1], z+2)]
                    new_height = np.min(neighborhood)
                    self.heightmap[x, z] = new_height

    # -----------------------------------------------------------
    # Public methods
    # -----------------------------------------------------------
    def load_world_slice(self):
        try:
            self.build_area = self.editor.getBuildArea()
        except Exception as e:
            print(f"Error getting build area: {e}")
            raise e
        
        self.worldslice = self.editor.loadWorldSlice(self.build_area.toRect(), cache=True)
        self.build_area_size = self.build_area.size
        self.build_area_offset = self.build_area.offset

        self.heightmap = self.worldslice.heightmaps[HEIGHTMAP_TYPE]
        
    def __compute_finite_difference_slope_scores(self):
        """ Compute the slope scores using finite difference"""
        if self.heightmap is None:
            raise ValueError("Heightmap not loaded. Call load_world_slice() first.")
        grad_x = np.zeros_like(self.heightmap)
        grad_z = np.zeros_like(self.heightmap)
        # Central difference for interior points
        grad_x[1:-1, :] = (self.heightmap[2:, :] - self.heightmap[:-2, :]) / 2
        grad_z[:, 1:-1] = (self.heightmap[:, 2:] - self.heightmap[:, :-2]) / 2
        # Forward/backward difference for boundary points
        grad_x[0, :] = self.heightmap[1, :] - self.heightmap[0, :]
        grad_x[-1, :] = self.heightmap[-1, :] - self.heightmap[-2, :]
        grad_z[:, 0] = self.heightmap[:, 1] - self.heightmap[:, 0]
        grad_z[:, -1] = self.heightmap[:, -1] - self.heightmap[:, -2]

        # Multiply by two to get integer slope scores
        grad_x *= 2
        grad_z *= 2

        # Compute the slope scores as the magnitude of the gradient
        self.slope_scores = grad_x.astype(int)**2 + grad_z.astype(int)**2


    def compute_slope_map(self, save=False):
        """
        Compute the slope map based on the heightmap, tree mask, and water mask. The slope scores are calculated using finite differences, and penalties are applied for water and tree blocks.
        """

        if self.worldslice is None:
            raise ValueError("World slice not loaded. Call load_world_slice() first.")
        
        self.__compute_tree_and_water_masks()
        self.__squash_trees_in_heightmap()
        self.__compute_finite_difference_slope_scores()

        height_x_size, height_z_size = self.heightmap.shape

        # Enforce penalty for water and tree blocks
        water_penalty = self.slope_scores.max() * WATER_PENALTY_FACTOR
        for x in range(height_x_size):
            for z in range(height_z_size):
                if self.water_mask[x, z]:
                    self.slope_scores[x, z] = water_penalty
                elif self.tree_mask[x, z]:
                    self.slope_scores[x, z] += TREE_PENALTY

        # Save to txt file
        if save:
            # Export masks to text files for debugging
            np.savetxt("./minecraft_gen/tree_mask.txt", self.tree_mask.astype(int), fmt="%d")
            np.savetxt("./minecraft_gen/water_mask.txt", self.water_mask.astype(int), fmt="%d")
            np.savetxt("./minecraft_gen/slope_scores.txt", self.slope_scores, fmt="%d")
            np.savetxt("./minecraft_gen/heightmap.txt", self.heightmap, fmt="%d")

    def load_world_map(self, heightmap_file, slope_scores_file, tree_mask_file, water_mask_file):
        """
        Load the world map data from text files. This includes the heightmap, slope scores, tree mask, and water mask.
        
        Args:
            heightmap_file (str): Path to the heightmap text file.
            slope_scores_file (str): Path to the slope scores text file.
            tree_mask_file (str): Path to the tree mask text file.
            water_mask_file (str): Path to the water mask text file.
        """
        self.heightmap = np.loadtxt(heightmap_file, dtype=int)
        self.slope_scores = np.loadtxt(slope_scores_file, dtype=int)
        self.tree_mask = np.loadtxt(tree_mask_file, dtype=bool)
        self.water_mask = np.loadtxt(water_mask_file, dtype=bool)        

    def place_block(self, local_coords, block: Block):
        """
        Place a block in the Minecraft world based on local coordinates within the build area.

        Args:
            local_coords (tuple): A tuple of (local_x, local_y, local_z) coordinates within the build area.
            block (Block): The block to place at the specified coordinates.
        """

        local_x, local_y, local_z = local_coords
        world_x = int(self.build_area_offset[X] + local_x)
        world_y = int(local_y)
        world_z = int(self.build_area_offset[Z] + local_z)
        self.editor.placeBlock((world_x, world_y, world_z), block)