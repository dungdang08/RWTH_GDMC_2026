import json
import os
import torch
from utils.log_utils import load_model
from model.fq_models import FQModel, HparamsFQGAN
from config.paths import BASE_LOG_DIR

# -------------------------------------------------------------------
# FQGAN HELPER FUNCTIONS
# -------------------------------------------------------------------
def load_hparams_from_json(log_dir):
    """
    Load hyperparameters from a JSON file located in the specified log directory.
    """

    log_dir = f"{BASE_LOG_DIR}/" + log_dir
    json_path = os.path.join(log_dir, 'hparams.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No hparams.json file found in {log_dir}")
    
    with open(json_path, 'r') as f:
        hparams = json.load(f)

    return hparams


def dict_to_vcqgan_hparams(hparams_dict, dataset=None):
    """
    Convert a dictionary of hyperparameters into an HparamsFQGAN object.
    """

    # Determine which hyperparameter class to use based on the dataset
    if dataset == None:
        dataset = hparams_dict.get('dataset', 'MNIST')  # Default to MNIST if not specified
    
    vq_hyper = HparamsFQGAN(dataset)
    # Set attributes from the dictionary
    for key, value in hparams_dict.items():
        setattr(vq_hyper, key, value)
    
    return vq_hyper


def load_fqgan_from_checkpoint(H, fqgan, model_name="fqgan", device='cuda'):
    """
    Load the FQGAN model weights from a checkpoint file specified in the hyperparameters.
    Args:
        H (HparamsFQGAN): Hyperparameters object containing the checkpoint information.
        fqgan (FQModel): The FQGAN model instance to load weights into.
        model_name (str): Name of the model for logging purposes.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    Returns:
        FQModel: The FQGAN model with loaded weights.
    """
    fqgan = load_model(fqgan, model_name, H.load_step, H.load_dir).to(device)
    fqgan.eval()
    return fqgan

def encode_and_quantize(fqgan, terrain_chunks, device='cuda'):
    """
    Memory-efficient encoding and quantization function. 
    Takes a chunk or batch of chunks from the dataset, returns the encoded style and structure indices matrices

    Args:
        fqgan (FQModel): The FQGAN model used for encoding and quantization.
        terrain_chunks (torch.Tensor): Input tensor of terrain chunks to encode.
        device (str): Device to perform computations on ('cuda' or 'cpu').
    Returns:
        tuple: A tuple containing the style indices and structure indices tensors.    
    """
    fqgan.eval()
    with torch.no_grad():
        # Move input to device
        terrain_chunks = terrain_chunks.to(device)
        
        # Get encodings
        h_style, h_struct = fqgan.encoder(terrain_chunks)
        
        # Process style path
        h_style = fqgan.quant_conv_style(h_style)
        quant_style, _, style_stats = fqgan.quantize_style(h_style)
        style_indices = style_stats[2]  # Get indices from tuple
        style_indices = style_indices.view(
            (h_style.size()[0], h_style.size()[2], h_style.size()[3], h_style.size()[4])
        )
        
        # Clear intermediate tensors
        del h_style, quant_style, style_stats
        
        # Process structure path
        h_struct = fqgan.quant_conv_struct(h_struct)
        quant_struct, _, struct_stats = fqgan.quantize_struct(h_struct)
        struct_indices = struct_stats[2]  # Get indices from tuple
        struct_indices = struct_indices.view(
            (h_struct.size()[0], h_struct.size()[2], h_struct.size()[3], h_struct.size()[4])
        )
        
        # Clear intermediate tensors
        del h_struct, quant_struct, struct_stats
        
        # Move indices to CPU to save GPU memory
        style_indices = style_indices.cpu()
        struct_indices = struct_indices.cpu()
        
        torch.cuda.empty_cache()
        
        return style_indices, struct_indices

# Takes style and structure indices, returns the reconstructed map
def decode_from_indices(style_indices, struct_indices, fqgan, device='cuda', two_stage=False, return_raw=False):
    """
    Decode the style and structure indices back into a reconstructed map using the FQGAN model.

    Args:
        style_indices (torch.Tensor): Tensor of style indices.
        struct_indices (torch.Tensor): Tensor of structure indices.
        fqgan (FQModel): The FQGAN model used for decoding.
        device (str): Device to perform computations on ('cuda' or 'cpu').
        two_stage (bool): Whether to use a two-stage decoding process.
        return_raw (bool): If True, return the raw decoded tensor without converting to block IDs.
    Returns:
        torch.Tensor: The reconstructed map tensor, or a tuple of (reconstructed map, binary decoded map) if two_stage is True.
    """
    with torch.no_grad():
        # Move indices to device only when needed
        style_indices = style_indices.to(device)
        struct_indices = struct_indices.to(device)
        
        # Get quantized vectors
        # quant_style = fqgan.quantize_style.get_codebook_entry(
        #     style_indices.view(-1),
        #     shape=[1, fqgan.embed_dim, *style_indices.shape[1:]]
        # )
        # quant_struct = fqgan.quantize_struct.get_codebook_entry(
        #     struct_indices.view(-1),
        #     shape=[1, fqgan.embed_dim, *struct_indices.shape[1:]]
        # )

        # Get quantized vectors from a batch
        batch_size = style_indices.shape[0]
        quant_style = fqgan.quantize_style.get_codebook_entry(
            style_indices.reshape(-1),
            shape=[batch_size, fqgan.embed_dim, *style_indices.shape[1:]]
        )
        quant_struct = fqgan.quantize_struct.get_codebook_entry(
            struct_indices.reshape(-1),
            shape=[batch_size, fqgan.embed_dim, *struct_indices.shape[1:]]
        )
        
        # Clear indices from GPU
        del style_indices, struct_indices
        
        # Combine and decode
        quant = torch.cat([quant_struct, quant_style], dim=1)
        # quant = quant_style + quant_struct
        del quant_style, quant_struct
        
        if two_stage:
            decoded, binary_decoded = fqgan.decoder(quant)
        else:
            decoded = fqgan.decoder(quant)
        
        del quant

        if return_raw:
            return decoded
        
        # Convert to block IDs if one-hot encoded
        if decoded.shape[1] > 1:
            decoded = torch.argmax(decoded, dim=1)
        
        # Move result to CPU and clear GPU memory
        result = decoded.squeeze(0).cpu()
        if two_stage:
            binary_result = binary_decoded.squeeze(0).cpu()
            del decoded
            del binary_decoded
            torch.cuda.empty_cache()
            return result, binary_result
        
        del decoded
        torch.cuda.empty_cache()
        
        return result
    
def get_model(model_name, model_path, model_checkpoint_step, device='cuda') -> tuple[FQModel, HparamsFQGAN]:
    """
    Loads the FQGAN model for a specific checkpoint step.

    Args:
        model_name (str): Name of the model to load.
        model_path (str): Path to the directory containing the model checkpoints.
        model_checkpoint_step (int): The specific checkpoint step to load.
        device (str): Device to load the model onto ('cuda' or 'cpu').
    Returns:
        tuple: A tuple containing the loaded FQGAN model and its hyperparameters.
    """
    try:
        fqgan_hparams =  dict_to_vcqgan_hparams(load_hparams_from_json(f"{model_path}"), 'minecraft')
        fqgan_hparams.load_step = model_checkpoint_step
        fqgan_hparams.load_dir = model_path
        fqgan = FQModel(fqgan_hparams)
        fqgan = load_fqgan_from_checkpoint(fqgan_hparams, fqgan, model_name=model_name, device=device)
        return fqgan, fqgan_hparams
    except Exception as e:
        print(f"Error loading model for step {model_checkpoint_step}: {e}")
        return None # Indicate failure