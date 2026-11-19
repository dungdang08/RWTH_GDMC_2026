import random
import torch
import pandas as pd
import os

from utils.data_utils import BlockBiomeConverter
from utils.model_utils import encode_and_quantize, decode_from_indices, get_model
from config.paths import DATA_PROCESSED_FOR_CONVERTER_DIR, DATA_BUILDINGS_DIR, DATA_PROCESSED_DIR, BASE_LOG_DIR


def generate_a_random_building(fqgan, sample_name, output_path, device='cuda'):
    """
    Generates a random building by encoding and decoding randomly selected style and structure indices from the dataset.
    Export the generated building as a CSV file with columns ['Position', 'Block'].

    Args:
        fqgan: The FQGAN model.
        sample_name (str): The name of the sample to use for randomly selected style and structure indices.
        output_path (str or Path): The path to save the generated building CSV file.
        device (str): The device to run the model on ('cuda' or 'cpu').
    """
    block_converter = BlockBiomeConverter.from_dataset(DATA_PROCESSED_FOR_CONVERTER_DIR / f"{sample_name}.npz")

    # Randomly combine style and structure indices from different samples
    sample = torch.load(DATA_PROCESSED_DIR / f'{sample_name}.pt')
    style_indices_list = []
    struct_indices_list = []
    for sample_chunk in sample["chunks"]:
        sample_chunk = sample_chunk.unsqueeze(0)  # keep batch dimension: [1, ...]
        with torch.no_grad():
            style_indices, struct_indices = encode_and_quantize(fqgan, sample_chunk, device=device)
        style_indices_list.append(style_indices)
        struct_indices_list.append(struct_indices) 

    random_style_indices = random.choice(style_indices_list)
    random_struct_indices = random.choice(struct_indices_list)
    
    reconstructed, _ = decode_from_indices(random_style_indices, random_struct_indices, fqgan, two_stage=True, device=device)
    # print(reconstructed)
    reconstruct_convert = block_converter.convert_to_block_names(reconstructed.squeeze())
    
    # Export to csv file
    block_entries = []
    for y in range(reconstruct_convert.shape[0]): # height
        for x in range(reconstruct_convert.shape[1]): # width
            for z in range(reconstruct_convert.shape[2]): # depth
                block_name = reconstruct_convert[y, x, z]
                if block_name not in ["minecraft:air"]:
                    block_entries.append(((x, y, z), block_name))
                # print(f"Block at (x={x}, y={y}, z={z}): Name={block_name}")
    # Save to CSV
    df = pd.DataFrame(block_entries, columns=["Position", "Block"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def generate_building_from_a_template(fqgan, template_name, output_path, device='cuda'):
    """
    Generates a building from a template by encoding and decoding the style and structure indices of the template.
    Export the generated building as a CSV file with columns ['Position', 'Block'].

    Args:
        fqgan: The FQGAN model.
        template_name (str): The name of the template to use for encoding and decoding.
        output_path (str or Path): The path to save the generated building CSV file.
        device (str): The device to run the model on ('cuda' or 'cpu').
    """
    sample = torch.load(DATA_PROCESSED_DIR / f'{template_name}.pt')
    block_converter = BlockBiomeConverter.from_dataset(DATA_PROCESSED_FOR_CONVERTER_DIR / f"{template_name}.npz")
    try:
        sample_chunk = sample["chunks"][0].unsqueeze(0)  # keep batch dimension: [1, ...]
        
        # Export to csv file
        sample_chunk_convert = torch.argmax(sample_chunk[0], dim=0)
        sample_chunk_convert = block_converter.convert_to_block_names(sample_chunk_convert.squeeze(0))
        block_entries = []
        for y in range(sample_chunk_convert.shape[0]): # height
            for x in range(sample_chunk_convert.shape[1]): # width
                for z in range(sample_chunk_convert.shape[2]): # depth
                    block_name = sample_chunk_convert[y, x, z]
                    if block_name not in ["minecraft:air", "air"]:
                        block_entries.append(((x, y, z), block_name))
                    # print(f"Block at (x={x}, y={y}, z={z}): Name={block_name}")
        # Save to CSV
        df = pd.DataFrame(block_entries, columns=["Position", "Block"])
        df.to_csv("template.csv", index=False)

        with torch.no_grad():
            style_indices, struct_indices = encode_and_quantize(fqgan, sample_chunk, device=device)
            print(f"Style indices: ", style_indices)
            print(f"Structure indices: ", struct_indices)
    except Exception as e:
        print(f"Error encoding sample {template_name}: {e}")
        return
    reconstructed, _ = decode_from_indices(style_indices, struct_indices, fqgan, two_stage=True, device=device)
    # print(reconstructed)
    reconstruct_convert = block_converter.convert_to_block_names(reconstructed.squeeze())
    
    # Export to csv file
    block_entries = []
    for y in range(reconstruct_convert.shape[0]): # height
        for x in range(reconstruct_convert.shape[1]): # width
            for z in range(reconstruct_convert.shape[2]): # depth
                block_name = reconstruct_convert[y, x, z]
                if block_name not in ["minecraft:air", "air"]:
                    block_entries.append(((x, y, z), block_name))
                # print(f"Block at (x={x}, y={y}, z={z}): Name={block_name}")
    # Save to CSV
    df = pd.DataFrame(block_entries, columns=["Position", "Block"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def generate_building_from_codes(fqgan, style_codes, struct_codes, output_path, device='cuda'):
    """
    Generates a building from given style and structure codes by decoding them using the FQGAN model.
    Export the generated building as a CSV file with columns ['Position', 'Block'].

    Args:
        fqgan: The FQGAN model.
        style_codes (torch.Tensor): The style codes to decode.
        struct_codes (torch.Tensor): The structure codes to decode.
        output_path (str or Path): The path to save the generated building CSV file.
        device (str): The device to run the model on ('cuda' or 'cpu').
    """
    block_converter = BlockBiomeConverter()
    # Check the dimensions of the style_code and struct_code
    if style_codes.shape != (1, 6, 6, 6) or struct_codes.shape != (1, 6, 6, 6):
        raise ValueError(f"Expected style_code and struct_code to have shape (1, 6, 6, 6), but got {style_codes.shape} and {struct_codes.shape}")

    reconstructed, _ = decode_from_indices(style_codes, struct_codes, fqgan, two_stage=True, device=device)
    # print(reconstructed)
    reconstruct_convert = block_converter.convert_to_block_names(reconstructed.squeeze())
    
    # Export to csv file
    block_entries = []
    for y in range(reconstruct_convert.shape[0]): # height
        for x in range(reconstruct_convert.shape[1]): # width
            for z in range(reconstruct_convert.shape[2]): # depth
                block_name = reconstruct_convert[y, x, z]
                if block_name not in ["minecraft:air", "air"]:
                    block_entries.append(((x, y, z), block_name))
                # print(f"Block at (x={x}, y={y}, z={z}): Name={block_name}")
    # Save to CSV
    df = pd.DataFrame(block_entries, columns=["Position", "Block"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "fqgan"
    model_dir = "logdir_64_64_3DCraft_full_grouped4"
    checkpoint_step = 2500

    fqgan, fqgan_hparams = get_model(model_name=model_name, model_path=model_dir, model_checkpoint_step=checkpoint_step, device=device)
    
    # template_name = "fhouse2"
    # output_path = DATA_BUILDINGS_DIR / "gdmc25-test-grouped2" / f"generated_building_from_{template_name}.csv"

    # template_name = "house9"
    # output_path = DATA_BUILDINGS_DIR / "3DCraft-test-grouped4" / f"generated_building_from_{template_name}.csv"
    # generate_building_from_a_template(fqgan, f"3DCraft_{template_name}_grouped4", output_path, device=device)

    # generate_a_random_building(fqgan, "24_3DCraft_grouped2_full", output_path, device=device)
    # generate_building_from_a_template(fqgan, f"3DCraft_{template_name}", output_path, device=device)
    
    style_codes = torch.tensor([[
         [[58, 58, 58, 58, 58, 58],
          [62, 62, 62, 62, 62, 62],
          [42, 42, 42, 42, 42, 42],
          [42, 42, 42, 42, 42, 42],
          [ 2,  2,  2,  2,  2, 2],
          [33, 33,  33, 33, 33, 33]],

         [[22, 22, 22, 22, 22, 22],
          [46, 46, 46, 46, 46, 46],
          [62, 62, 62, 62, 62, 62],
          [56, 56, 56, 56, 56, 56],
          [32, 32, 32, 32, 32, 32],
          [48, 48, 48, 48, 48, 48]],

         [[62, 62, 62, 62, 62, 62],
          [42, 42, 42, 42, 42, 42],
          [41, 41, 41, 41, 41, 41],
          [55, 55, 55, 55, 55,  55],
          [43, 43, 43, 43, 43, 43],
          [6, 6, 6, 6, 6, 6]],

         [[22, 22, 22, 22, 22, 22],
          [23, 23, 23, 23, 23, 23],
          [18, 18, 18, 18, 18, 18],
          [23, 23, 23, 23, 23, 23],
          [32, 32, 32, 32, 32, 32],
          [50, 50, 50, 50, 50, 50]],

         [[19, 19, 19, 19, 19, 19],
          [13, 13, 13, 13, 13, 13],
          [ 4,  4,  4,  4,  4,  4],
          [ 1,  1,  1,  1,  1,  1],
          [50, 50, 50, 50, 50, 50],
          [17, 17, 17, 17, 17, 17]],

         [[43, 43, 43, 43, 43, 43],
          [22, 22, 22, 22, 22, 22],
          [45, 45, 45, 45, 45, 45],
          [ 1,  1,  1,  1,  1,  1],
          [43, 43, 43, 43, 43, 43],
          [17, 17, 17, 17, 17, 17]]]])
    
    struct_codes = torch.tensor([[
         [[56, 56, 56, 56, 56, 56],
          [0,  0,  0,  0,  0,  0],
          [0,  0,  0,  0,  0,  0],
          [17, 17, 17, 17, 17, 17],
          [44, 44, 44, 44, 44, 44],
          [44, 44, 44, 44, 44, 44]],

         [[56, 56, 56, 56, 56, 56],
          [56, 56, 56, 56, 56, 56],
          [56,  56, 56, 56, 56, 56],
          [3,  3,  3,  3,  3, 3],
          [33, 33, 33, 33, 33, 33],
          [50, 50, 50, 50, 50, 50]],

         [[56, 56, 56, 56, 56, 56],
          [19, 19, 19, 19,  19, 19],
          [17, 17, 17, 17, 17, 17],
          [17, 17, 17, 17, 17, 17],
          [30, 30, 30, 30, 30, 30],
          [23, 23, 23, 23, 23, 23]],

         [[58, 58, 58, 58, 58, 58],
          [ 3,  3,  3,  3,  3,  3],
          [3,  3,  3,  3,  3,  3],
          [ 3,  3,  3,  3,  3,  3],
          [25, 25, 25, 25, 25, 25],
          [25, 25, 25, 25, 25, 25]],

         [[ 4,  4,  4,  4,  4,  4],
          [17, 17, 17, 17, 17, 17],
          [17, 17, 17, 17, 17, 17],
          [31, 31, 31, 31, 31, 31],
          [46, 46, 46, 46, 46, 46],
          [25, 25, 25, 25, 25, 25]],

         [[ 4,  4,  4,  4,  4,  4],
          [37, 37, 37, 37, 37, 37],
          [56, 56, 56, 56, 56, 56],
          [33, 33, 33, 33, 33, 33],
          [50, 50, 50, 50, 50, 50],
          [43, 43, 43, 43, 43, 43]]]])
    
    output_path = DATA_BUILDINGS_DIR / "gdmc25-test-grouped4" / f"generated_building_from_codes2.csv"
    generate_building_from_codes(fqgan, style_codes, struct_codes, output_path, device=device)