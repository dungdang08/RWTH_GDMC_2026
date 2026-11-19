import numpy as np
import nbtlib
import torch.nn.functional as F
import torch
import json
from pathlib import Path

from utils.data_utils import BlockBiomeConverter
from utils.model_utils import encode_and_quantize, decode_from_indices, get_model
from config.paths import DATA_DIR, BLOCK_INDEX_TO_NAME_DICT_PATH

class HouseGenerator:
    MODEL_NAME = "fqgan"
    MODEL_DIR = "logdir_64_64_3DCraft_full_grouped4"
    CHECKPOINT_STEP = 2500
    INPUT_CHUNK_SIZE = 24

    def __read_schem(self, input_schem_path, chunk_size=INPUT_CHUNK_SIZE) -> tuple[np.ndarray, dict]:
        """
        Read a Minecraft schematic file and convert it to a voxel representation.

        Args:
            input_schem_path (str or Path): Path to the input .schem file.
            chunk_size (int): The size of the voxel chunk to create. Default is 24.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the voxel array and a mapping of block indices to block names.

        """
        nbt = nbtlib.load(input_schem_path)
        schem = nbt["Schematic"]

        width = schem["Width"]
        height = schem["Height"]
        depth = schem["Length"]

        padded = False

        if width > chunk_size or height > chunk_size or depth > chunk_size:
            raise ValueError(f"Schematic dimensions ({width}x{height}x{depth}) exceed chunk size ({chunk_size}x{chunk_size}x{chunk_size}).")

        if width != chunk_size or height != chunk_size or depth != chunk_size:
            padded = True
            print(f"Warning: Schematic dimensions ({height}x{width}x{depth}) (HxWxD) are smaller than chunk size ({chunk_size}x{chunk_size}x{chunk_size}). "
                f"Output will be padded to chunk size with air.")

        blocks_compound = schem["Blocks"]  # a compound tag containing "Palette" and "Data"
        palette = blocks_compound["Palette"]  # a mapping of block names to indices
        block_data = blocks_compound["Data"]  # a flat array of block indices corresponding to the palette

        print(f"Schematic dimensions: Width={width}, Height={height}, Length={depth}")
        print(f"Palette size: {len(palette)} unique blocks")

        voxel_shape = (height, width, depth)  # Minecraft uses Y as height, X as width, and Z as depth
        voxels = np.zeros(voxel_shape, dtype=np.int32)
        
        block_types = {}  # Mapping from palette index to block name
        for block_name, tag in palette.items():
            # Extract block name without properties
            block_types[tag] = block_name.split("[")[0]  
        # print(f"Block types mapping: {block_types}")

        for y in range(height):
            for x in range(width):
                for z in range(depth):
                    index = y * depth * width + z * width + x
                    block_index = block_data[index]  # This is the index in the palette
                    voxels[y, x, z] = block_index

        # Get the index for air from the palette, default to 0 if not found
        air_index = palette.get("minecraft:air")
        if air_index is None:
            air_index = len(palette)
            palette["minecraft:air"] = air_index
            block_types[air_index] = "minecraft:air"
        # Pad the voxel array with air
        if padded:
            padded_voxels = np.full((chunk_size, chunk_size, chunk_size), air_index)
            padded_voxels[:height, :width, :depth] = voxels
            voxels = padded_voxels
        return voxels, block_types


    def __group_blocks(self, voxels, block_types, group_dict_name="group_dict") -> tuple[np.ndarray, dict]:
        """
        Group blocks in the voxel array based on a specified grouping dictionary.

        Args:
            voxels (np.ndarray): The voxel array containing block indices.
            block_types (dict): A mapping of block indices to block names.
            group_dict_name (str): The name of the grouping dictionary to use.

        Returns:
            tuple[np.ndarray, dict]: A tuple containing the grouped voxel array and a mapping of group indices to group names.
        """
        with open(DATA_DIR / f"{group_dict_name}.json", "r") as f:
            group_dict = json.load(f)
            group_dict = {group.lower(): [block_name.lower() for block_name in block_names] for group, block_names in group_dict.items()}

        group_to_index = {group: idx for idx, group in enumerate(group_dict.keys())}
        index_to_group = {idx: group for group, idx in group_to_index.items()}

        grouped_voxels = np.full_like(voxels, fill_value=group_to_index["air"])
        for raw_idx, raw_block_name in block_types.items():
            raw_idx = int(raw_idx)
            raw_block_name = raw_block_name.lower().replace("minecraft:", "")
            group_name = "air"
            for group, block_names in group_dict.items():
                if raw_block_name in block_names:
                    # print(f"DEBUG: Mapping raw block '{raw_block_name}' (index {raw_idx}) to group '{group}'")
                    group_name = group
                    break
            group_idx = group_to_index[group_name]
            # Assign the group index to all voxels with corresponding raw block index
            grouped_voxels[voxels.astype(int) == int(raw_idx)] = group_idx
        return grouped_voxels, index_to_group


    def __convert_raw_voxels_to_chunks(self, voxels, block_types, block_name_to_index, group_dict_name=""):
        if group_dict_name:
            voxels, block_types = self.__group_blocks(voxels, block_types, group_dict_name)

        voxel_chunk = np.zeros_like(voxels, dtype=np.int32)
        for palette_index, block_name in block_types.items():
            global_index = block_name_to_index.get(block_name, -1)  # Default to -1 if not found
            if global_index == -1:
                # Get air index if block name is not found
                if "minecraft:air" in block_name_to_index.keys():
                    global_index = block_name_to_index["minecraft:air"]
                elif "air" in block_name_to_index.keys():
                    global_index = block_name_to_index["air"]
                else:
                    global_index = 0  # Default to 0 if air is not found

            voxel_chunk[voxels == palette_index] = global_index
        return voxel_chunk
    

    def __generate_building_from_codes(self, fqgan, style_codes, struct_codes, device='cuda'):
        """
        Generate a building from the given style and structure codes using the FQGAN model.
        
        Args:
            fqgan: The FQGAN model.
            style_codes: The style codes tensor.
            struct_codes: The structure codes tensor.
            device: The device to run the model on ('cuda' or 'cpu').
        """

        block_converter = BlockBiomeConverter()
        # Check the dimensions of the style_code and struct_code
        if style_codes.shape != (1, 6, 6, 6) or struct_codes.shape != (1, 6, 6, 6):
            raise ValueError(f"Expected style_code and struct_code to have shape (1, 6, 6, 6), but got {style_codes.shape} and {struct_codes.shape}")

        reconstructed, _ = decode_from_indices(style_codes, struct_codes, fqgan, two_stage=True, device=device)
        # print("DEBUG:", reconstructed)
        reconstruct_convert = block_converter.convert_to_block_names(reconstructed.squeeze())
        
        block_entries = []
        for y in range(reconstruct_convert.shape[0]): # height
            for x in range(reconstruct_convert.shape[1]): # width
                for z in range(reconstruct_convert.shape[2]): # depth
                    block_name = reconstruct_convert[y, x, z]
                    if block_name not in ["minecraft:air", "air"]:
                        block_entries.append(((x, y, z), block_name))
                    # print(f"Block at (x={x}, y={y}, z={z}): Name={block_name}")
        return block_entries

    def __encode_building(self, fqgan, fqgan_params, voxel_chunk, device='cuda'):
        """
        Encode a voxel chunk into style and structure codes using the FQGAN model.

        Args:
            fqgan: The FQGAN model.
            fqgan_params: The parameters of the FQGAN model.
            voxel_chunk: The voxel chunk to encode.
            device: The device to run the model on ('cuda' or 'cpu').
        Returns:
            tuple: A tuple containing the style codes and structure codes.
        """

        chunks_index = torch.from_numpy(voxel_chunk).long().unsqueeze(0)
        num_block_types = fqgan_params.n_channels
        chunks_one_hot = F.one_hot(chunks_index, num_classes=num_block_types,).permute(0, 4, 1, 2, 3).float()
    
        # [N, 24, 24, 24, C] -> [N, C, 24, 24, 24]
        style_codes, struct_codes = encode_and_quantize(chunks_one_hot, fqgan, device=device)
        
        return style_codes, struct_codes

    def __extract_info_from_buiding(self, block_entries):
        if not block_entries:
            raise ValueError("The reconstructed building contains no non-air blocks.")
        
        min_x = min(entry[0][0] for entry in block_entries)
        max_x = max(entry[0][0] for entry in block_entries)
        min_y = min(entry[0][1] for entry in block_entries)
        max_y = max(entry[0][1] for entry in block_entries)
        min_z = min(entry[0][2] for entry in block_entries)
        max_z = max(entry[0][2] for entry in block_entries)

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        depth = max_z - min_z + 1
        y_offset = min_y

        # Get door position
        door_positions = [entry[0] for entry in block_entries if "door" in entry[1].lower()]
        if door_positions:
            door_x, _, door_z = door_positions[0]
            door_position = (door_x - min_x, door_z - min_z)
        else:
            door_position = None
            # print("No door found in the building.")
        return width, height, depth, y_offset, door_position

    def generate_buildings_from_templates(self, template_path_dict):
        """
        Generate buildings from a dictionary of template paths and the number of buildings to generate for each template.
        
        Args:
            template_path_dict (dict): A dictionary where keys are template paths and values are the number of buildings to generate for each template.

        Returns:
            list: A list of dictionaries containing building information, including size, door position, and block entries.
        """
        building_types = []
        generated_buildings = []
        if BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
            with open(BLOCK_INDEX_TO_NAME_DICT_PATH, "r") as f:
                block_index_to_name = json.load(f)
                block_name_to_index = {block_name: int(idx) for idx, block_name in block_index_to_name.items()}
        else:
            raise FileNotFoundError(f"Block index to name dictionary not found at {BLOCK_INDEX_TO_NAME_DICT_PATH}.")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fqgan, fqgan_hparams = get_model(model_name=self.MODEL_NAME,
                                          model_path=self.MODEL_DIR, 
                                          model_checkpoint_step=self.CHECKPOINT_STEP, 
                                          device=device)
        fqgan.eval()

        for template_path, num_buildings in template_path_dict.items():
            template_path = Path(template_path)
            voxels, block_types = self.__read_schem(template_path)
            voxel_chunk = self.__convert_raw_voxels_to_chunks(voxels, block_types, block_name_to_index, group_dict_name="group_dict4")
            with torch.inference_mode():
                style_codes, struct_codes = self.__encode_building(fqgan, fqgan_hparams, voxel_chunk, device=device)
                block_entries = self.__generate_building_from_codes(fqgan, style_codes, struct_codes, device=device)
                building_width, building_height, building_depth, building_y_offset, door_position = self.__extract_info_from_buiding(block_entries)
                if door_position is None:
                    print(f"Warning: No door found in building template {template_path}. Setting door position to (0, 0).")
                    door_position = (building_width // 2, 0) # Default door position if none found
                building_types.append({
                    'name': template_path.stem,
                    'size': (building_width, building_height, building_depth),
                    'max_num': num_buildings,
                    'door_pos': door_position,
                    'border': max(building_width, building_depth) // 5,
                    'y_offset': building_y_offset,
                    'blocks': block_entries
                })
                generated_buildings.append(block_entries)

        return building_types

        
        

    