# Source: https://github.com/TimMerino1710/DisentanglingMinecraft/blob/master/data_utils.py
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import torch
# import nbtlib
# import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import json

from torch.utils.data import Dataset, DataLoader
import random

from config.paths import META_BLOCK_INDEX_TO_NAME_DICT_PATH, BLOCK_INDEX_TO_NAME_DICT_PATH


def combine_chunk_files(input_directory, output_file_path):
    """
    Combines multiple Minecraft chunk NPZ files from a directory into a single NPZ file.
    
    Args:
        input_directory (str or Path): Path to directory containing NPZ files with voxels and biome data
        output_file_path (str or Path): Path where the combined data will be saved
    
    Returns:
        dict: Information about the combined data including counts and shapes
    """
    input_directory = Path(input_directory)
    output_file_path = Path(output_file_path)
    
    # Ensure input directory exists
    if not input_directory.exists() or not input_directory.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_directory}")
    
    # Find all NPZ files in the directory
    npz_files = list(input_directory.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in directory: {input_directory}")
    
    print(f"Found {len(npz_files)} NPZ files in {input_directory}")
    
    # Load the first file to determine shapes
    first_data = np.load(npz_files[0], allow_pickle=True)
    voxel_shape = first_data['voxels'].shape
    biome_shape = first_data['biome'].shape if 'biome' in first_data else None
    
    # Prepare arrays to hold all data
    all_voxels = []
    all_biomes = []
    
    # Process all files
    for i, file_path in enumerate(npz_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(npz_files)}")
        
        try:
            data = np.load(file_path, allow_pickle=True)
            voxels = data['voxels']
            
            # Verify shape consistency
            if voxels.shape != voxel_shape:
                print(f"Warning: Skipping file {file_path} due to shape mismatch. "
                      f"Expected {voxel_shape}, got {voxels.shape}")
                continue
            
            all_voxels.append(voxels)
            
            # Handle biomes if present
            if 'biome' in data:
                biomes = data['biome']
                if biome_shape is None:
                    biome_shape = biomes.shape
                if biomes.shape != biome_shape:
                    print(f"Warning: Biome shape mismatch in {file_path}. "
                          f"Expected {biome_shape}, got {biomes.shape}")
                all_biomes.append(biomes)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Stack all data
    combined_voxels = np.stack(all_voxels)
    combined_biomes = np.stack(all_biomes) if all_biomes else None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_file_path.parent, exist_ok=True)
    
    # Save combined data
    if combined_biomes is not None:
        np.savez(output_file_path, chunks=combined_voxels, biomes=combined_biomes)
        print(f"Saved combined data to {output_file_path}")
        print(f"Combined voxel shape: {combined_voxels.shape}")
        print(f"Combined biome shape: {combined_biomes.shape}")
        return {
            "num_chunks": len(all_voxels),
            "voxel_shape": combined_voxels.shape,
            "biome_shape": combined_biomes.shape
        }
    else:
        np.savez(output_file_path, chunks=combined_voxels)
        print(f"Saved combined voxel data to {output_file_path} (no biome data found)")
        print(f"Combined voxel shape: {combined_voxels.shape}")
        return {
            "num_chunks": len(all_voxels),
            "voxel_shape": combined_voxels.shape,
            "biome_shape": None
        }
    
    
def rotate_voxels_90(voxels, k=1):
    """
    Rotate voxels around Y axis by k*90 degrees
    Args:
        voxels: tensor of shape [B, C, H, W, D] or [C, H, W, D]
        k: number of 90 degree rotations (1 = 90°, 2 = 180°, 3 = 270°)
    Returns:
        Rotated voxels
    """
    # Handle both batched and unbatched inputs
    if len(voxels.shape) == 5:  # Batched [B, C, H, W, D]
        # Rotate around Y (height) axis by swapping width and depth dimensions
        return torch.rot90(voxels, k=k, dims=(2, 4))
    elif len(voxels.shape) == 4:  # Unbatched [C, H, W, D]
        return torch.rot90(voxels, k=k, dims=(1, 3))
    elif len(voxels.shape) == 3:  # Unbatched [H, W, D]
        return torch.rot90(voxels, k=k, dims=(0, 2))
    else:
        raise ValueError(f"Unexpected voxel shape: {voxels.shape}")
    
class MinecraftRotationAugmentation:
    def __init__(self, p=0.75):
        """
        Args:
            p: probability of applying rotation (0-1)
        """
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            # Randomly choose rotation: 90°, 180°, or 270°
            k = random.randint(1, 3)
            return rotate_voxels_90(x, k)
        return x

class MinecraftDataset(Dataset):
    def __init__(self, data_path, augment=True, rotation_prob=0.75):
        data_path = Path(data_path)
        self.augment = augment
        self.rotation_prob = rotation_prob
        # Try to load processed data first

        print("Loading pre-processed data...")
        processed_data = torch.load(data_path)
        # Only keep the chunks, discard biome data
        self.processed_chunks = processed_data['chunks']
        # Delete the biomes to free memory
        del processed_data['biomes']
        del processed_data
        
        
        print(f"Loaded {len(self.processed_chunks)} chunks of size {self.processed_chunks.shape[1:]}")
        print(f"Number of unique block types: {self.processed_chunks.shape[1]}")
        # print(f'Unique blocks: {torch.unique(torch.argmax(self.processed_chunks, dim=1)).tolist()}')

    def __getitem__(self, idx):
        chunk = self.processed_chunks[idx]
        
        # Apply random rotation augmentation during training
        if self.augment and random.random() < self.rotation_prob:
            # Randomly choose rotation: 90°, 180°, or 270°
            k = random.randint(1, 3)
            chunk = rotate_voxels_90(chunk, k)
            
        return chunk
    
    def __len__(self):
        return len(self.processed_chunks)

def get_minecraft_dataloaders(data_path, batch_size=32, val_split=0.1, num_workers=0, save_val_path=None, augment=True, device='cpu'):
    """
    Creates training and validation dataloaders for Minecraft chunks.
    """
    # Create dataset
    dataset = MinecraftDataset(data_path)
    
    # Split into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    val_dataset.dataset.augment = False

    # Save validation data if path provided
    if save_val_path:
        print(f'saving validation dataset to file: {save_val_path}')
        # Extract validation samples
        val_samples = torch.stack([dataset.processed_chunks[i] for i in val_dataset.indices])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_val_path), exist_ok=True)
        
        # Save validation data
        torch.save({
            'data': val_samples,
            'indices': val_dataset.indices
        }, save_val_path)
        print(f"Saved validation data to {save_val_path}")
    
    # Create dataloaders with memory pinning
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False,
    )

    
    print(f"\nDataloader details:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader



# So that our conversion back to the original indexes isn't tied to the dataset, we can just load it from file
class BlockConverter:
    def __init__(self, index_to_block_map=None, block_to_index_map=None):
        """
        Initialize with pre-computed mappings
        Can be initialized either with saved mappings or by creating new ones
        """
        self.index_to_block = index_to_block_map
        self.block_to_index = block_to_index_map
    
    @classmethod
    def from_dataset(cls, data_path):
        """Create mappings from a dataset file"""
        chunks = torch.from_numpy(np.load(data_path))
        unique_blocks = torch.unique(chunks).numpy()
        
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        return cls(index_to_block, block_to_index)
    
    @classmethod
    def load_mappings(cls, path):
        """Load pre-saved mappings"""
        mappings = torch.load(path)
        return cls(mappings['index_to_block'], mappings['block_to_index'])
    
    def save_mappings(self, path):
        """Save mappings for later use"""
        torch.save({
            'index_to_block': self.index_to_block,
            'block_to_index': self.block_to_index
        }, path)
    
    def convert_to_original_blocks(self, data):
        """Convert from indices back to original block IDs"""
        if len(data.shape) == 5:  # [B, C, H, W, D] or [C, H, W, D]
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        return torch.tensor([[[[self.index_to_block[int(b)] 
                              for b in row]
                             for row in layer]
                            for layer in slice_]
                           for slice_ in data])
    
class BlockBiomeConverter:
    def __init__(self, block_mappings=None, biome_mappings=None):
        """
        Initialize with pre-computed mappings for both blocks and biomes
        
        Args:
            block_mappings: dict containing 'index_to_block' and 'block_to_index'
            biome_mappings: dict containing 'index_to_biome' and 'biome_to_index'
        """
        self.block_to_str =  {}
        if META_BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
            loaded_block_to_str = json.load(open(BLOCK_INDEX_TO_NAME_DICT_PATH, 'r'))
            self.block_to_str = {int(k): v for k, v in loaded_block_to_str.items()}
        elif BLOCK_INDEX_TO_NAME_DICT_PATH.exists():
            loaded_block_to_str = json.load(open(BLOCK_INDEX_TO_NAME_DICT_PATH, 'r'))
            self.block_to_str = {int(k): v for k, v in loaded_block_to_str.items()}
        else:
            raise ValueError("No block index to name dictionary found")
            
        self.index_to_block = block_mappings['index_to_block'] if block_mappings else None
        self.block_to_index = block_mappings['block_to_index'] if block_mappings else None
        self.index_to_biome = biome_mappings['index_to_biome'] if biome_mappings else None
        self.biome_to_index = biome_mappings['biome_to_index'] if biome_mappings else None
    
    @classmethod
    def from_dataset(cls, data_path):
        """Create mappings from a dataset file"""
        data = np.load(data_path, allow_pickle=True)
        voxels = data['voxels']
        biomes = data['biomes']
        
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}
        
        block_mappings = {'index_to_block': index_to_block, 'block_to_index': block_to_index}
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def from_arrays(cls, voxels, biomes):
        """Create mappings directly from numpy arrays"""
        # Create block mappings (blocks are integers)
        unique_blocks = np.unique(voxels)
        block_to_index = {int(block): idx for idx, block in enumerate(unique_blocks)}
        index_to_block = {idx: int(block) for idx, block in enumerate(unique_blocks)}
        
        # Create biome mappings (biomes are strings)
        unique_biomes = np.unique(biomes)
        biome_to_index = {str(biome): idx for idx, biome in enumerate(unique_biomes)}
        index_to_biome = {idx: str(biome) for idx, biome in enumerate(unique_biomes)}
        
        block_mappings = {'index_to_block': index_to_block, 'block_to_index': block_to_index}
        biome_mappings = {'index_to_biome': index_to_biome, 'biome_to_index': biome_to_index}
        
        return cls(block_mappings, biome_mappings)
    
    @classmethod
    def load_mappings(cls, path):
        """Load pre-saved mappings"""
        mappings = torch.load(path)
        return cls(mappings['block_mappings'], mappings['biome_mappings'])
    
    def save_mappings(self, path):
        """Save mappings for later use"""
        torch.save({
            'block_mappings': {
                'index_to_block': self.index_to_block,
                'block_to_index': self.block_to_index
            },
            'biome_mappings': {
                'index_to_biome': self.index_to_biome,
                'biome_to_index': self.biome_to_index
            }
        }, path)
    
    def convert_to_original_blocks(self, data):
        """
        Convert from indices back to original block IDs.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_blocks), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original blocks
        if len(data.shape) == 4:  # Batch dimension present
            return torch.tensor([[[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in slice_]
                                for slice_ in data])
        else:  # No batch dimension
            return torch.tensor([[[self.index_to_block[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in data])
        
    def convert_to_indices(self, data):
        """
        Convert from indices back to original block IDs.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
        Returns:
            torch.Tensor of original block IDs with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_blocks), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original blocks
        if len(data.shape) == 4:  # Batch dimension present
            return torch.tensor([[[[self.block_to_index[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in slice_]
                                for slice_ in data])
        else:  # No batch dimension
            return torch.tensor([[[self.block_to_index[int(b)] 
                                for b in row]
                                for row in layer]
                                for layer in data])

    def convert_to_original_biomes(self, data):
        """
        Convert from indices back to original biome strings.
        Handles both one-hot encoded and already-indexed data.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded biomes [B, C, H, W, D] or [C, H, W, D]
                - indexed biomes [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of original biome strings with shape [B, H, W, D] or [H, W, D]
        """
        # If one-hot encoded (dim == 5 or first dim == num_biomes), convert to indices first
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.biome_to_index)):
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
        
        # Now convert indices to original biomes
        if len(data.shape) == 4:  # Batch dimension present
            return np.array([[[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.index_to_biome[int(b)] 
                            for b in row]
                            for row in layer]
                            for layer in data])

    def get_block_name_from_index(self, index):
        """
        Convert a single index to the corresponding block name.
        
        Args:
            index: int - the index of the block type
        Returns:
            str - the name of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        block_id = self.index_to_block[index]
        return self.block_to_str.get(block_id, f"UNKNOWN_BLOCK_{block_id}")
    
    def get_block_id_from_index(self, index):
        """
        Convert a single index to the corresponding block ID.
        
        Args:
            index: int - the index of the block type
        Returns:
            int - the ID of the block
        """
        if self.index_to_block is None:
            raise ValueError("Block mappings not initialized")
        
        return self.index_to_block[index]
    
    def convert_to_block_names(self, data):
        """
        Convert from indices or IDs to block names.
        
        Args:
            data: torch.Tensor of either:
                - one-hot encoded blocks [B, C, H, W, D] or [C, H, W, D]
                - indexed blocks [B, H, W, D] or [H, W, D]
                - block ID blocks [B, H, W, D] or [H, W, D]
        Returns:
            numpy array of block names with shape [B, H, W, D] or [H, W, D]
        """
        # First convert to block IDs if necessary
        if len(data.shape) == 5 or (len(data.shape) == 4 and data.shape[0] == len(self.block_to_index)):
            # One-hot encoded, convert to indices first
            data = torch.argmax(data, dim=1 if len(data.shape) == 5 else 0)
            # Then convert indices to block IDs
            data = self.convert_to_original_blocks(data)
        elif self.index_to_block is not None and (
            (len(data.shape) == 4 and torch.max(data) < len(self.index_to_block)) or
            (len(data.shape) == 3 and torch.max(data) < len(self.index_to_block))
        ):
            # These are indices, convert to block IDs
            data = self.convert_to_original_blocks(data)
        
        # Now data contains block IDs, convert to names
        if data.dim() == 4:  # Batch dimension present
            return np.array([[[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in slice_]
                            for slice_ in data])
        else:  # No batch dimension
            return np.array([[[self.block_to_str.get(int(b), f"UNKNOWN_BLOCK_{int(b)}") 
                            for b in row]
                            for row in layer]
                            for layer in data])
        
    def get_air_block_index(self):
        """
        Find the one-hot index corresponding to the air block (ID 0).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 5 (air) in our index_to_block mapping
        for idx, block_id in self.index_to_block.items():
            if block_id == 0:  # Air block ID
                return idx
        raise ValueError("Air block (ID 0) not found in block mappings!")
    
    def get_water_block_index(self):
        """
        Find the one-hot index corresponding to the air block (ID 5).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 240 (air) in our index_to_block mapping
        for idx, block_id in self.index_to_block.items():
            if block_id == 240 :  # water block ID
                return idx
        raise ValueError("water block (ID 240) not found in block mappings!")
    
    def get_blockid_indices(self, block_ids):
        """
        Find the one-hot index corresponding to the air block (ID 0).
        Returns:
            int: The index where air blocks are encoded in one-hot format
        """
        # Find the index that maps to block ID 5 (air) in our index_to_block mapping
        idxs = []
        for idx, block_id in self.index_to_block.items():
            if block_id in block_ids:  # Air block ID
                idxs.append(idx)
        if len(idxs) == 0:
            raise ValueError("Air block (ID 0) not found in block mappings!")
        return idxs
    
    def get_important_blockname_indices(self, block_names):
        """
        Find the one-hot indices corresponding to a list of (partly) block names (excluding air and bedrock).
        Args:
            block_names: list of str - the block names to search for
        Returns:
            list of int: The indices where the specified blocks are encoded in one-hot format
        """
        idxs = []
        for idx, block_id in self.index_to_block.items():
            block_name = self.block_to_str.get(block_id, f"UNKNOWN_BLOCK_{block_id}")
            stripped_block_name = block_name.replace("minecraft:", "")
            if stripped_block_name == "bedrock" or stripped_block_name == "air":
                continue  # Skip bedrock
            if stripped_block_name in block_names:
                idxs.append(idx)
        if len(idxs) == 0:
            raise ValueError(f"No blocks found matching names: {block_names}")
        return idxs