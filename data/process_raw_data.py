# This script converts raw Minecraft schematic files into a standardized voxel format for training the model. 

import os
import nbtlib
import numpy as np
import json
import torch


from utils.data_utils import BlockBiomeConverter
from config.paths import DATA_DIR, DATA_PROCESSED_FOR_CONVERTER_DIR, DATA_PROCESSED_DIR, \
                         GDMC_2025_RAW_DATA_DIR, HOUSE_3D_CRAFT_DATA_DIR, HOUSE_3D_CRAFT_DATA_DIR_GABLE_ROOF, HOUSE_3D_CRAFT_DATA_DIR_FILTERED, DATA_BUILDINGS_DIR, \
                         BLOCK_INDEX_TO_NAME_DICT_PATH, BIOME_INDEX_TO_NAME_DICT_PATH


CHUNK_SIZE = 24
BIOME = "buildings" # Placeholder architectural style name, can be updated based on actual data


# --------------------------------------------------------------------------------
# GDMC 2025 dataset processing functions
# --------------------------------------------------------------------------------
def read_schem(input_schem_path, chunk_size=16) -> tuple[np.ndarray, dict]:
    """
    Return a 3D numpy array of block indices and a mapping of palette indices to block names from a .schematic file.
    If the schematic is smaller than the chunk size, it will be padded with air.
    If the schematic is larger than the chunk size, a ValueError will be raised.
    
    Args:
        input_schem_path (str or Path): Path to the .schematic file
        chunk_size (int): Maximum allowed dimension for width, height, and length
    
    Returns:
        voxels (np.ndarray): A 3D array of shape (chunk_size, chunk_size, chunk_size) containing block indices
        block_types (dict): A mapping from palette index to block name
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
    voxels = np.zeros(voxel_shape, dtype=np.float32)
    
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
    air_index = palette.get("minecraft:air", -1)
    if air_index == -1:
        print("Warning: 'minecraft:air' not found in palette. Adding air to palette.")
        palette["minecraft:air"] = len(palette)
    # Pad the voxel array with air
    if padded:
        padded_voxels = np.full((chunk_size, chunk_size, chunk_size), air_index)
        padded_voxels[:height, :width, :depth] = voxels
        voxels = padded_voxels

    return voxels, block_types


def read_all_schems_in_directory(directory_path, filter_house=True, chunk_size=16) -> tuple[list[np.ndarray], list[dict]]:
    """
    Read all .schematic files in the specified directory and return a list of voxel arrays and block type mappings.
    
    Args:
        directory_path (str or Path): Path to the directory containing .schematic files
        filter_house (bool): If True, only process schematics with "house" in the filename
        chunk_size (int): Maximum allowed dimension for width, height, and length
        
    Returns:
        combined_voxels (list of np.ndarray): A list of 3D arrays containing block indices for each schematic
        combined_block_types (list of dict): A list of mappings from palette index to block name for each schematic
    """
    schem_files = [f for f in os.listdir(directory_path) if f.endswith(".schem")]
    
    combined_voxels = []
    combined_block_types = []

    for schem_file in schem_files:
        if filter_house:
            if "house" not in schem_file.lower():
                print(f"Skipping {schem_file} as it does not appear to be a house schematic.")
                continue
        input_schematic_path = directory_path / schem_file
        print(f"Processing {schem_file}...")
        try:
            voxels, block_type_dict = read_schem(input_schematic_path, chunk_size=chunk_size)
            combined_voxels.append(voxels)
            combined_block_types.append(block_type_dict)
        except Exception as e:
            print(f"Error processing {input_schematic_path}: {e}")
        print("\n")
    
    return combined_voxels, combined_block_types



# --------------------------------------------------------------------------------
# 3D-Craft dataset processing functions
# --------------------------------------------------------------------------------
def read_3DCraft_npy_file(input_folder, chunk_size=24) -> tuple[np.ndarray, dict]:
    """
    Read the 3D Craft .npy file and return a 3D numpy array of block indices and a mapping of palette indices to block name.

    Args:
        input_folder (str or Path): Path to the folder containing "schematic.npy" and "stats.json"
    Returns:
        voxels (np.ndarray): A 3D array of shape (X (Width), Y (Height), Z (Depth)) containing block indices
        block_types (dict): A mapping from palette index to block name

    """
    try:
        data = np.load(input_folder / "schematic.npy", allow_pickle=True) # a 4d numpy array in (y, z, x, id/meta) with id/meta = 2-tuple of ints: (block id, block meta)
        stats = json.load(open(input_folder / "stats.json", "r")) # aggregate statistics, e.g. schematic size,
    except Exception as e:
        print(f"Error reading 3D Craft dataset files in {input_folder}: {e}")
        raise
    
    y, z, x = stats["size"]
    voxels = np.empty((y, x, z), dtype=object)  # Initialize voxel array with shape (Y, X, Z)
    block_types_list = []
    
    print(f"DEBUG: voxels shape of house folder {os.path.basename(input_folder)}: {voxels.shape}")    
    if y > chunk_size or z > chunk_size or x > chunk_size:
        raise ValueError(f"Voxel dimensions ({voxels.shape}) exceed chunk size ({chunk_size}, {chunk_size}, {chunk_size}). Skipping this house.")
    
    # Load dict mapping pre-flattening block indices to block names from the JSON file
    with open(DATA_DIR / "legacy_112_to_121_dict.json", "r") as f:
        legacy_112_to_121_dict = json.load(f)
    
    for y in range(data.shape[0]):
        for z in range(data.shape[1]):
            for x in range(data.shape[2]):
                block_id, block_meta = data[y, z, x]
                pre_flattening_index = f"{block_id}:{block_meta}" if int(block_meta) != 0 else f"{block_id}"
                block_name = legacy_112_to_121_dict.get(pre_flattening_index, "minecraft:air")
                voxels[y, x, z] = block_name  # Note the order of x and z to match Minecraft's coordinate system
                block_types_list.append(block_name)

    unique_block_types = set(block_types_list)
    block_name_to_index = {
        block_name: idx
        for idx, block_name in enumerate(unique_block_types)
    }
    block_types = {idx: block_name for block_name, idx in block_name_to_index.items()}
    voxels = np.vectorize(block_name_to_index.get)(voxels)

    # Pad the voxel array with air
    air_index = -1
    if "minecraft:air" in block_types_list:
        air_index = next(idx for idx, name in block_types.items() if name == "minecraft:air")
    if air_index == -1:
        print("DEBUG: 'minecraft:air' not found in block types. Adding air to block types.")
        air_index = len(block_types)
        block_types[air_index] = "minecraft:air"
    padded_voxels = np.full((chunk_size, chunk_size, chunk_size), air_index)
    padded_voxels[:voxels.shape[0], :voxels.shape[1], :voxels.shape[2]] = voxels
    voxels = padded_voxels

    return voxels, block_types               


def read_all_3DCraft_dataset(directory_path, chunk_size: None | int = 24, max_size=None) -> tuple[list[np.ndarray], list[dict]]:
    """
    Read all 3D Craft dataset folders in the specified directory and return a list of voxel arrays and block type mappings.
    
    Args:
        directory_path (str or Path): Path to the directory containing 3D Craft dataset folders
        chunk_size (int): Maximum allowed dimension for width, height, and length
        max_size (int): Maximum number of houses to read from the dataset. If None, read all houses.
    Returns:
        combined_voxels (list of np.ndarray): A list of 3D arrays containing block indices for each house
        combined_block_types (list of dict): A list of mappings from palette index to block name for each house
    """
    all_house_folders = [f for f in os.listdir(directory_path) if f.startswith("workdir")]
    # rand_house_folders = np.random.choice(all_house_folders, size=min(max_size, len(all_house_folders)), replace=False)
    if max_size is not None:
        house_folders = all_house_folders[:min(max_size, len(all_house_folders))]
    else:
        house_folders = all_house_folders
    combined_voxels = []
    combined_block_types = []
    for house_folder in house_folders:
        # print(f"Processing {house_folder}...")
        try:
            voxels, block_types = read_3DCraft_npy_file(directory_path / house_folder, chunk_size=chunk_size)
        except Exception as e:
            print(f"Error processing {house_folder}: {e}")
            continue
        combined_voxels.append(voxels)
        combined_block_types.append(block_types)
    return combined_voxels, combined_block_types


# --------------------------------------------------------------------------------
# Block grouping functions for 3D Craft dataset
# --------------------------------------------------------------------------------
def group_blocks(combined_voxels, combined_block_types, group_dict_name="group_dict", threshold=0.9) -> tuple[list[np.ndarray], list[dict]]:
    """
    Group blocks in the voxel arrays based on a specified grouping dictionary and filter out voxel arrays that mostly contain air blocks.
    
    Args:
        combined_voxels (list of np.ndarray): A list of 3D arrays containing block indices for each house
        combined_block_types (list of dict): A list of mappings from palette index to block name for each house
        group_dict_name (str): The name of the JSON file containing the block grouping dictionary (without .json extension)
        threshold (float): The maximum allowed ratio of ignored blocks (e.g., air) in a voxel array for it to be considered valid.
    Returns:
        valid_grouped_voxels (list of np.ndarray): A list of 3D arrays containing grouped block indices for valid houses
        grouped_block_types (list of dict): A list of mappings from group index to group name for valid houses
    """
    valid_grouped_voxels = []
    grouped_block_types = []

    # Read group_dict from the JSON file
    # with open(DATA_DIR / "group_dict.json", "r") as f:
    #     group_dict = json.load(f)
    #     group_dict = {group.lower(): [block_name.lower() for block_name in block_names] for group, block_names in group_dict.items()}

    with open(DATA_DIR / f"{group_dict_name}.json", "r") as f:
        group_dict = json.load(f)
        group_dict = {group.lower(): [block_name.lower() for block_name in block_names] for group, block_names in group_dict.items()}

    group_to_index = {group: idx for idx, group in enumerate(group_dict.keys())}
    index_to_group = {idx: group for group, idx in group_to_index.items()}

    for house_idx, voxels in enumerate(combined_voxels):
        grouped_voxels = np.full_like(voxels, fill_value=group_to_index["air"])
        block_dict = combined_block_types[house_idx]
        for raw_idx, raw_block_name in block_dict.items():
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
        
        # Remove voxel arrays that mostly contain AIR, IGNORE_SOLID or FLOOR blocks
        total_blocks = grouped_voxels.size        
        air_blocks = np.sum(grouped_voxels == group_to_index["air"])

        if air_blocks / total_blocks < threshold:
            valid_grouped_voxels.append(grouped_voxels)
            print(f"Valid house index {house_idx} with {air_blocks / total_blocks:.2%} ignored blocks.")
        else:
            print(f"Skipping house index {house_idx} with {air_blocks / total_blocks:.2%} ignored blocks.")
            pass

    grouped_block_types = [index_to_group.copy() for _ in range(len(valid_grouped_voxels))]
    print("DEBUG: Grouped block types mapping:", grouped_block_types[0])
    return valid_grouped_voxels, grouped_block_types


# --------------------------------------------------------------------------------
# 3D-Craft dataset helper functions
# --------------------------------------------------------------------------------
def export_3DCraft_blocktype_dict(directory_path, output_json_path, max_size=None):
    """
    Export a dictionary mapping block indices to block names for the 3D Craft dataset as a JSON file.
    
    Args:
        directory_path (str or Path): Path to the directory containing 3D Craft dataset folders
        output_json_path (str or Path): Path to the output JSON file
        max_size (int): Maximum number of houses to read from the dataset. If None, read all houses.
    """
    house_folders = [f for f in os.listdir(directory_path) if f.startswith("workdir")]
    if max_size is not None:
        house_folders = house_folders[:max_size]
    unique_block_types = set()
    for house_folder in house_folders:
        print(f"Processing {house_folder} for unique block types...")
        try:
            ignore, block_types = read_3DCraft_npy_file(directory_path / house_folder)
            unique_block_types.update(block_types.values())
        except Exception as e:
            # print(f"Error processing {house_folder}")
            continue
    unique_block_types.discard("minecraft:air")
    unique_block_types = ["minecraft:air"] + sorted(unique_block_types)
    block_name_to_index = {
        idx: str(block_name)
        for idx, block_name in enumerate(unique_block_types)
    }
    with open(output_json_path, "w") as f:
        json.dump(block_name_to_index, f, indent=4)

def export_voxel_array_to_csv(voxel_array, block_index_to_name, output_csv_path):
    """
    Export a 3D voxel array to a CSV file with columns ['Position', 'Block'].
    Args:
        voxel_array (np.ndarray): A 3D array of shape (X, Y, Z) containing block indices
        block_index_to_name (dict): A mapping from block index to block name
        output_csv_path (str or Path): Path to the output CSV file
    """
    import pandas as pd

    block_entries = []
    for y in range(voxel_array.shape[1]): # height
        for x in range(voxel_array.shape[0]): # width
            for z in range(voxel_array.shape[2]): # depth
                block_index = int(voxel_array[x, y, z])
                block_name = block_index_to_name.get(block_index, "unknown_block")
                if block_name not in ["minecraft:air", "minecraft:grass_block", "minecraft:dirt"]:
                    block_entries.append(((x, y, z), block_name))

    df = pd.DataFrame(block_entries, columns=["Position", "Block"])
    df.to_csv(output_csv_path, index=False)

# Test function
def test_read_3DCraft_npy_file(test_input_folder, test_output_csv_path):
    """
    Test function to read a sample 3D Craft .npy file and export it to CSV for verification.
    """
    voxels, block_types = read_3DCraft_npy_file(test_input_folder)
    print(f"Unique block types: {block_types}")
    export_voxel_array_to_csv(voxels, block_types, test_output_csv_path)
    print(f"Exported voxel array to {test_output_csv_path}")


# --------------------------------------------------------------------------------
#                                   MAIN FUNCTION 
# --------------------------------------------------------------------------------
def main():
    os.makedirs(DATA_PROCESSED_FOR_CONVERTER_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    group_block = True
    # --------------------------------------------------------------------------------
    # Get all .schem files in the GDMC_2025_RAW_DATA_DIR

    # all_voxels_gdmc, all_block_types_gdmc = read_all_schems_in_directory(GDMC_2025_RAW_DATA_DIR, filter_house=True, chunk_size=CHUNK_SIZE)
    
    # --------------------------------------------------------------------------------
    # Get all houses in the 3DCraft dataset directory
    # NOTE:
    # Download data from https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz 
    # and place the unzipped folders in the HOUSE_3D_CRAFT_DATA_DIR before running this script
    
    # max_dataset_size = None
    # all_voxels, all_block_types = read_all_3DCraft_dataset(HOUSE_3D_CRAFT_DATA_DIR, chunk_size=CHUNK_SIZE, max_size=max_dataset_size)
    # all_voxels, all_block_types = read_all_3DCraft_dataset(HOUSE_3D_CRAFT_DATA_DIR_GABLE_ROOF, chunk_size=CHUNK_SIZE, max_size=max_dataset_size)
    # --------------------------------------------------------------------------------
    # Combine the GDMC and 3D Craft datasets

    # all_voxels = all_voxels_gdmc + all_voxels
    # all_block_types = all_block_types_gdmc + all_block_types

    # --------------------------------------------------------------------------------
    # Get one house from the gdmc 2025 dataset as template for testing
    # template_name = "fhouse2"
    # voxels, block_types = read_schem(GDMC_2025_RAW_DATA_DIR / f"{template_name}.schem", chunk_size=CHUNK_SIZE)
    # all_voxels, all_block_types = [voxels], [block_types]

    # --------------------------------------------------------------------------------
    # Get one house from the 3D-Craft dataset as template for testing
    template_name = "house9"
    voxels, block_types = read_3DCraft_npy_file(HOUSE_3D_CRAFT_DATA_DIR_FILTERED / f"workdir.{template_name}", chunk_size=CHUNK_SIZE)
    all_voxels, all_block_types = [voxels], [block_types]


    if group_block:
        all_voxels, all_block_types = group_blocks(all_voxels, all_block_types, group_dict_name="group_dict4", threshold=0.98)
    
    print("Finished reading dataset. Number of valid houses: ", len(all_voxels))
    if len(all_voxels) == 0:
        print("No valid voxel data found in the dataset")
        return
    
    # --------------------------------------------------------------------------------
    # Create a global mapping of block types across all schematics

    # Load the block index to name mapping from the JSON file
    if BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
        with open(BLOCK_INDEX_TO_NAME_DICT_PATH, "r") as f:
            block_index_to_name = json.load(f)
            block_name_to_index = {block_name: int(idx) for idx, block_name in block_index_to_name.items()}
    else:
        unique_block_types = set([block_name for block_dict in all_block_types for block_name in block_dict.values()])
        # Assign index 0 to air and create a mapping for the rest of the block types
        air_block_name = "minecraft:air" if "minecraft:air" in unique_block_types else "air"
           
        unique_block_types.discard(air_block_name)
        unique_block_types = [air_block_name] + sorted(unique_block_types)

        block_name_to_index = {
            str(block_name): idx
            for idx, block_name in enumerate(unique_block_types)
        }

        block_index_to_name = {
            idx: str(block_name)
            for block_name, idx in block_name_to_index.items()
        }
        
        print(f"Total unique block types across all schematics: {len(unique_block_types)}")
        # print(f"Unique block types: {block_index_to_name}")
        # Export block index to name mapping as a JSON file for reference
    
        with open(BLOCK_INDEX_TO_NAME_DICT_PATH, "w") as f:
            json.dump(block_index_to_name, f, indent=4)

    if BIOME_INDEX_TO_NAME_DICT_PATH.exists():
        with open(BIOME_INDEX_TO_NAME_DICT_PATH, "r") as f:
            biome_index_to_name = json.load(f)
            biome_name_to_index = {str(block_name): int(idx) for idx, block_name in biome_index_to_name.items()}
    else:
        biome_index_to_name = {int(0): BIOME,}
        biome_name_to_index = {BIOME: int(0),}  # Placeholder biome index for the architectural style
        # Export biome index to name mapping as a JSON file for reference
        with open(BIOME_INDEX_TO_NAME_DICT_PATH, "w") as f:
            json.dump(biome_index_to_name, f, indent=4)
    print("DEBUG: Block name to index mapping:", block_name_to_index)
    # --------------------------------------------------------------------------------
    # Remap voxel block indices to global indices based on the block name to index mapping

    # Assign new indices to the voxel arrays based on the global block name to index mapping
    for i in range(len(all_voxels)):
        if i >= len(all_block_types):
            print(f"DEBUG: No block type dictionary for schematic index {i}. Skipping index remapping for this schematic.")
            continue
        block_type_dict = all_block_types[i]
        try:
            palette_index_to_global_index = {
                int(palette_index): block_name_to_index[block_name]
                for palette_index, block_name in block_type_dict.items()
            }
        except KeyError as e:
            print(f"Error: Block name {e} from schematic index {i} not found in global block name to index mapping. This block will be skipped in the remapping process.")
            palette_index_to_global_index = {
                int(palette_index): block_name_to_index.get(block_name, block_name_to_index["minecraft:air"])
                for palette_index, block_name in block_type_dict.items()
            }
        # Update the voxel array with global indices
        voxels = all_voxels[i]
        vectorized_mapping = np.vectorize(palette_index_to_global_index.get)
        all_voxels[i] = vectorized_mapping(voxels)
        # if i == 0:
        #     print("DEBUG: palette_index_to_global_index mapping:", palette_index_to_global_index)
    # ------------------------------------------------------------------------------------------------------------------------
    # Save the all voxels and biomes into a compressed dataset file (*.npz) and save the mappings into a .pt file

    all_data_file_name = f"3DCraft_{template_name}_grouped4"
    # all_data_file_name = f"{CHUNK_SIZE}_3DCraft_grouped4_full"
    combined_raw_data_file_path = DATA_PROCESSED_FOR_CONVERTER_DIR / f"{all_data_file_name}.npz"
    combined_processed_data_file_path = DATA_PROCESSED_DIR / f"{all_data_file_name}.pt"
    combined_processed_mapping_file_path = DATA_PROCESSED_DIR / f"{all_data_file_name}_mappings.pt"
    
    combined_voxels = np.stack(all_voxels)
    combined_biomes = np.array([biome_name_to_index[BIOME]] * len(all_voxels))
    
    np.savez_compressed(combined_raw_data_file_path, voxels=combined_voxels, biomes=combined_biomes)

    # Save block mappings for later use in the model
    block_biome_converter = BlockBiomeConverter.from_dataset(combined_raw_data_file_path)
    # print(f"Block mappings: {block_biome_converter.index_to_block}")
    # print(f"Biome mappings: {block_biome_converter.index_to_biome}")
    block_biome_converter.save_mappings(combined_processed_mapping_file_path)

    # ------------------------------------------------------------------------------------------------------------------------
    # Save the combined voxel data in a .pt file for loading into the model during training/validation/testing

    # Save chunks into pt files
    import torch.nn.functional as F

    chunks_index = torch.from_numpy(combined_voxels).long()
    num_block_types = len(block_name_to_index)
    print("combined_voxels min:", combined_voxels.min())
    print("combined_voxels max:", combined_voxels.max())
    print("unique voxel values:", np.unique(combined_voxels))
    print(f"Number of block types (including air): {num_block_types}")

    chunks_one_hot = F.one_hot(chunks_index, num_classes=num_block_types)
    chunks_one_hot = chunks_one_hot.permute(0, 4, 1, 2, 3).float()
    # [N, 24, 24, 24, C] -> [N, C, 24, 24, 24]

    torch.save({
        "chunks": chunks_one_hot,
        "biomes": torch.from_numpy(combined_biomes).long(),
    }, combined_processed_data_file_path)


if __name__ == "__main__":
    main()
    
    # export_3DCraft_blocktype_dict(HOUSE_3D_CRAFT_DATA_DIR, DATA_DIR / "block_index_to_name.json", max_size=300)
    # test_files = [f for f in os.listdir(HOUSE_3D_CRAFT_DATA_DIR_GABLE_ROOF) if f.startswith("workdir")]
    # for idx, test_file in enumerate(test_files):
    #     print(f"Testing read_3DCraft_npy_file with {test_file}...")
    #     test_read_3DCraft_npy_file(HOUSE_3D_CRAFT_DATA_DIR_GABLE_ROOF / test_file, DATA_BUILDINGS_DIR / f"3dcraft_sample_output{idx+1}.csv")
   
