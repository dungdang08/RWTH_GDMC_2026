# Source: https://github.com/TimMerino1710/DisentanglingMinecraft/blob/master/train.py

import numpy as np
import torch
import os
import time
# import random
# import visdom

from utils.data_utils import BlockBiomeConverter, get_minecraft_dataloaders
from utils.log_utils import log, save_model, save_stats, config_log, start_training_log, log_hparams_to_json, load_model
from model.fq_models import FQModel, VQLossDualCodebook, HparamsFQGAN
from config.paths import NEW_DATASET_PATH, NEW_MAPPINGS_PATH, BASE_LOG_DIR


def load_pretrained_vae_into_vqgan(vqgan, pretrained_vae_name, hparams, step, log_dir, device='cpu', strict=False):
    """
    Load a pretrained VAE into the VQGAN model, excluding the quantizer parameters.
    
    Args:
        vqgan (FQModel): The VQGAN model to load the pretrained weights into.
        pretrained_vae_name (str): The name of the pretrained VAE model.
        hparams (HparamsFQGAN): Hyperparameters for the VQGAN model.
        step (int): The training step at which to load the pretrained weights.
        log_dir (str): The directory where the pretrained model is logged.
        device (str): The device to load the model onto ('cpu' or 'cuda').
        strict (bool): Whether to strictly enforce that the keys in state_dict match the model's keys.
    """

    pretrained_vae = FQModel(hparams).to(device)
    pretrained_vae = load_model(pretrained_vae, pretrained_vae_name, step, log_dir, strict=strict).to(device)
    pretrained_state_dict = pretrained_vae.state_dict()
    vqgan_state_dict = vqgan.state_dict()

    # Exclude quantizer parameters 
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'quantize' not in k}
    updated_state_dict = {**vqgan_state_dict, **pretrained_state_dict}
    vqgan.load_state_dict(updated_state_dict)

    print("Pretrained VAE weights loaded into VQGAN successfully.")

@torch.no_grad()
def initialize_fq_codebooks_from_encoder(vqgan, train_loader, device, num_batches=20):
    """
    Initialize style and structure codebooks from the current encoder outputs.

    Args:
        vqgan (FQModel): The VQGAN model containing the encoder and codebooks.
        train_loader (DataLoader): DataLoader for the training dataset.
        device (str): Device to perform computations on ('cpu' or 'cuda').
        num_batches (int): Number of batches to use for initialization.
    """
    vqgan.eval()

    style_vectors = []
    struct_vectors = []

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        x = batch[0] if isinstance(batch, list) else batch
        x = x.to(device)

        h_style, h_struct = vqgan.encoder(x)

        h_style = vqgan.quant_conv_style(h_style)
        h_struct = vqgan.quant_conv_struct(h_struct)

        # [B, C, H, W, D] -> [B * H * W * D, C]
        h_style = h_style.permute(0, 2, 3, 4, 1).reshape(-1, h_style.shape[1])
        h_struct = h_struct.permute(0, 2, 3, 4, 1).reshape(-1, h_struct.shape[1])

        style_vectors.append(h_style.detach())
        struct_vectors.append(h_struct.detach())

    if len(style_vectors) == 0 or len(struct_vectors) == 0:
        raise RuntimeError("No latent vectors collected for codebook initialization.")

    style_vectors = torch.cat(style_vectors, dim=0)
    struct_vectors = torch.cat(struct_vectors, dim=0)

    print(f"Collected style vectors: {style_vectors.shape}")
    print(f"Collected struct vectors: {struct_vectors.shape}")
    
    try:
        style_codebook_size = vqgan.quantize_style.n_e
        struct_codebook_size = vqgan.quantize_struct.n_e
    except AttributeError:
        raise AttributeError("Style/Struct quantizer does not have n_e attribute.")

    if style_vectors.shape[0] < style_codebook_size or struct_vectors.shape[0] < struct_codebook_size:
        raise RuntimeError(f"Not enough style/struct vectors. Got {style_vectors.shape[0]}/{struct_vectors.shape[0]}, need {style_codebook_size}/{struct_codebook_size}.")

    style_perm = torch.randperm(style_vectors.shape[0], device=device)
    struct_perm = torch.randperm(struct_vectors.shape[0], device=device)

    style_init = style_vectors[style_perm[:style_codebook_size]].clone()
    struct_init = struct_vectors[struct_perm[:struct_codebook_size]].clone()

    if getattr(vqgan.quantize_style, "l2_norm", False):
        style_init = torch.nn.functional.normalize(style_init, p=2, dim=-1)

    if getattr(vqgan.quantize_struct, "l2_norm", False):
        struct_init = torch.nn.functional.normalize(struct_init, p=2, dim=-1)

    def copy_init_to_quantizer(quantizer, init_vectors, name):
        if not hasattr(quantizer, "embedding"):
            raise AttributeError(f"{name} quantizer has no embedding attribute.")

        # Standard VQ: nn.Embedding
        if isinstance(quantizer.embedding, torch.nn.Embedding):
            if quantizer.embedding.weight.shape != init_vectors.shape:
                raise RuntimeError(f"{name} codebook shape mismatch: {quantizer.embedding.weight.shape} vs {init_vectors.shape}")
            init_vectors = init_vectors.to(device=quantizer.embedding.weight.device, dtype=quantizer.embedding.weight.dtype)
            quantizer.embedding.weight.data.copy_(init_vectors)

        # EMA VQ: tensor buffer
        elif torch.is_tensor(quantizer.embedding):
            if quantizer.embedding.shape != init_vectors.shape:
                raise RuntimeError(f"{name} codebook shape mismatch: {quantizer.embedding.shape} vs {init_vectors.shape}")
            init_vectors = init_vectors.to(device=quantizer.embedding.device, dtype=quantizer.embedding.dtype)
            quantizer.embedding.data.copy_(init_vectors)

            if hasattr(quantizer, "embedding_avg"):
                quantizer.embedding_avg.data.copy_(init_vectors)

            if hasattr(quantizer, "cluster_size"):
                quantizer.cluster_size.data.fill_(1.0)

        else:
            raise TypeError(f"Unsupported embedding type for {name}: {type(quantizer.embedding)}")

        if hasattr(quantizer, "codebook_used"):
            quantizer.codebook_used.data.zero_()

    copy_init_to_quantizer(vqgan.quantize_style, style_init, "style")
    copy_init_to_quantizer(vqgan.quantize_struct, struct_init, "struct")

    vqgan.train()

    print("Initialized FQ codebooks from pretrained encoder latents.")
    print(f"Style active initialization size: {style_codebook_size}")
    print(f"Struct active initialization size: {struct_codebook_size}")

    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = HparamsFQGAN(dataset='minecraft')
    H.log_dir = 'logdir_64_64_3DCraft_full_grouped4'
    H.load_dir = 'loaddir_64_64_3DCraft_full_grouped4'
    log_model_name = "fqgan"
    log_optimizer_g_name = "optimizer_g"
    log_optimizer_d_name = "optimizer_d"

    # vis = visdom.Visdom(port=H.visdom_port)
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for VQGAN on {H.dataset}')

    start_training_log(H)
    log_hparams_to_json(H, H.log_dir)

    data_path = NEW_DATASET_PATH
    mappings_path = NEW_MAPPINGS_PATH
    # visualizer = MinecraftVisualizerPyVista() # No visualization during training for now
    print("Loading dataset and creating dataloaders...")
    train_loader, val_loader = get_minecraft_dataloaders(
        data_path,
        batch_size=H.batch_size,
        num_workers=0,
        val_split=0.05,
        augment=True,
        device=device,
        # vert=True
        # save_val_path=f'../../text2env/data/{H.log_dir}_valset.pt'
    )
    block_converter = BlockBiomeConverter.load_mappings(mappings_path)
    air_idx = block_converter.get_air_block_index()
    # water_idx = block_converter.get_water_block_index() # 
    print(f'setting air idx in hparams to {air_idx}')
    H.air_class_index = air_idx
    
    important_blocks = [
        "door",
    ]
    weight_block_idxs = block_converter.get_important_blockname_indices(important_blocks)
    print(f'log indices: {weight_block_idxs}')
    H.weighted_block_indices = weight_block_idxs
    # H.weighted_block_indices = []

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    # Initialize model and loss
    vqgan = FQModel(H).to(device)
    if H.load_pretrained_vae:
        print(f"Using pretrained AE/VAE initialization: {H.pretrained_vae_path}")
        load_pretrained_vae_into_vqgan(vqgan, f'{log_model_name}_pre', H, 2000, 'logdir_32_32_pretrained_w-o_weighted', device=device)
        # initialize_fq_codebooks_from_encoder(vqgan, train_loader, device, num_batches=20)
    vq_loss = VQLossDualCodebook(H).to(device)

    train_iterator = cycle(train_loader)
    if val_loader is not None:
        val_iterator = cycle(val_loader)

    # Setup optimizers 
    optimizer_g = torch.optim.Adam(
        vqgan.parameters(), 
        lr=H.lr,
        betas=(H.beta1, H.beta2),
        weight_decay=H.weight_decay
    )
    optimizer_d = torch.optim.Adam(
        vq_loss.discriminator.parameters(),
        lr=H.lr,
        betas=(H.beta1, H.beta2),
        weight_decay=H.weight_decay
    )

    if H.load_model:
        print(f"Loading model from {H.load_dir}...")
        vqgan = load_model(vqgan, log_model_name, H.start_step, H.log_dir, strict=False).to(device)
        optimizer_g = load_model(optimizer_g, log_optimizer_g_name, H.start_step, H.log_dir)
        optimizer_d = load_model(optimizer_d, log_optimizer_d_name, H.start_step, H.log_dir)
        print(f"Model loaded from {H.log_dir} at step {H.start_step}.")

    print(f'Using weight decay {H.weight_decay}')
    # Initialize loss tracking
    plotted_loss_terms = [
        'rec_loss', 'binary_rec_loss', 'style_loss', 'struct_loss', 
        'biome_feat_loss', 'disent_loss', 'g_loss', 'd_loss', 'struct_consistency_loss', 'cycle_consistency_loss'
    ]
    loss_arrays = {term: np.array([]) for term in plotted_loss_terms}
    val_loss_arrays = {term: np.array([]) for term in plotted_loss_terms}
    best_overall_val_loss = float('inf')
    codebook_usage = {
        'style': np.array([]),
        'struct': np.array([])
    }

    # Training loop
    epochs_without_improvement = 0  # Counter for early stopping

    step_start_time = time.time()
    print(f"\n--- Step {H.start_step + 1}/{H.train_steps} --- (Time: {time.strftime('%H:%M:%S', time.gmtime(step_start_time))}) ---")
    for step in range(H.start_step + 1, H.train_steps + 1):
        if step % 10 == 0:  # Print time every 10 steps
            step_start_time = time.time()
            print(f"\n--- Step {step}/{H.train_steps} --- (Time: {time.strftime('%H:%M:%S', time.gmtime(step_start_time))}) ---")
        batch = next(train_iterator)
        
        if isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        x = x.to(device)

        
        
        binary_target = (torch.argmax(x, dim=1) != air_idx).float().to(device)
        if H.two_stage_decoder:
            recons, binary_out, codebook_loss_style, codebook_loss_struct, disentangle_loss, biome_feat, struct_consistency, cycle_consistency = vqgan(x)
        else:
            recons, codebook_loss_style, codebook_loss_struct, disentangle_loss, biome_feat, struct_consistency, cycle_consistency = vqgan(x)
            binary_out = None

        
        # Generator/Encoder update
        optimizer_g.zero_grad()
        # Calculate generator losses
        loss_dict = vq_loss(
            codebook_loss_style, codebook_loss_struct,
            x, recons, disentangle_loss, biome_feat,
            optimizer_idx=0,
            global_step=step,
            binary_out=binary_out,
            binary_target=binary_target,
            struct_consistency_loss = struct_consistency,
            cycle_consistency_loss=cycle_consistency
        )
            
            
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(vqgan.parameters(), max_norm=1.0)
        optimizer_g.step()

        
        d_loss_dict = {}

        # Discriminator update
        if step >= H.disc_start_step:
            optimizer_d.zero_grad()
            
            # Calculate discriminator losses
            d_loss_dict = vq_loss(
                codebook_loss_style, codebook_loss_struct,
                x, recons, disentangle_loss, biome_feat,
                optimizer_idx=1,
                global_step=step,
                binary_out=binary_out,
                binary_target=binary_target
            )

            # print(f"Final d_loss: {d_loss_dict}")        d_loss_dict['d_loss'].backward()
            torch.nn.utils.clip_grad_norm_(vq_loss.discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

        # Logging
        if step % H.steps_per_log == 0:
            print(f"Logging at step {step}:")
            for term in plotted_loss_terms:
                if term in loss_dict:
                    value = loss_dict[term].item() if torch.is_tensor(loss_dict[term]) else loss_dict[term]
                    loss_arrays[term] = np.append(loss_arrays[term], value)
                    print(f"{term}: {value:.4f}")
                elif term == 'd_loss':
                    if step >= H.disc_start_step:
                        value = d_loss_dict['d_loss'].item() if torch.is_tensor(d_loss_dict['d_loss']) else d_loss_dict['d_loss']
                    else:
                        value = 0.0
                    loss_arrays[term] = np.append(loss_arrays[term], value)
                    print(f"{term}: {value:.4f}")

            # Update codebook usage tracking
            style_usage = loss_dict['codebook_usage_style'] if torch.is_tensor(loss_dict['codebook_usage_style']) else loss_dict['codebook_usage_style']
            struct_usage = loss_dict['codebook_usage_struct'] if torch.is_tensor(loss_dict['codebook_usage_struct']) else loss_dict['codebook_usage_struct']
            
            codebook_usage['style'] = np.append(codebook_usage['style'], style_usage)
            codebook_usage['struct'] = np.append(codebook_usage['struct'], struct_usage)
            print(f"Codebook Usage - Style: {(style_usage*100):.2f}%, Structure: {(struct_usage*100):.2f}%")
        
        # Add validation evaluation
        if step % H.steps_per_validation == 0:
            num_val_batches = 0
            if val_loader is not None:
                num_val_batches = getattr(H, 'val_num_batches', 4)
                num_val_batches = min(num_val_batches, len(val_loader))  
            if num_val_batches == 0:
                print("No validation batches available, skipping validation.")
            else:
                print(f"\nValidation at step {step} with number of batches {num_val_batches}:")

                vqgan.eval()
                vq_loss.eval()

                with torch.no_grad():
                    collected_val_losses = {term: [] for term in plotted_loss_terms}
                    sum_val_loss = 0.0
                    for _ in range(num_val_batches):
                        val_batch = next(val_iterator)
                        if isinstance(val_batch, list):
                            val_x = val_batch[0]
                        else:
                            val_x = val_batch
                        val_x = val_x.to(device)

                        val_binary_target = (torch.argmax(val_x, dim=1) != air_idx).float().to(device)
                        if H.two_stage_decoder:
                            val_recons, val_binary_out, val_codebook_loss_style, val_codebook_loss_struct, val_disentangle_loss, val_biome_feat, val_struct_consistency, val_cycle_consistency = vqgan(val_x)
                        else:
                            val_recons, val_codebook_loss_style, val_codebook_loss_struct, val_disentangle_loss, val_biome_feat, val_struct_consistency, val_cycle_consistency = vqgan(val_x)
                            val_binary_out = None

                        # Calculate generator losses
                        val_loss_dict = vq_loss(
                            val_codebook_loss_style, val_codebook_loss_struct,
                            val_x, val_recons, val_disentangle_loss, val_biome_feat,
                            optimizer_idx=0,
                            global_step=step,
                            binary_out=val_binary_out,
                            binary_target=val_binary_target,
                            struct_consistency_loss = val_struct_consistency,
                            cycle_consistency_loss=val_cycle_consistency
                        )

                        for term in plotted_loss_terms:
                            if term in val_loss_dict:
                                value = val_loss_dict[term].item() if torch.is_tensor(val_loss_dict[term]) else val_loss_dict[term]
                                collected_val_losses[term].append(value)
                        sum_val_loss += val_loss_dict['loss'].item() if torch.is_tensor(val_loss_dict['loss']) else val_loss_dict['loss']
                    
                    for term in plotted_loss_terms:
                        if len(collected_val_losses[term]) > 0:
                            val_value = np.mean(collected_val_losses[term])
                            val_loss_arrays[term] = np.append(val_loss_arrays[term], val_value)
                            print(f"Validation {term}: {val_value:.4f}")
                    
                    # Update best validation loss
                    avg_val_loss = sum_val_loss / num_val_batches
                    if avg_val_loss < best_overall_val_loss - H.early_stopping_min_delta:
                        epochs_without_improvement = 0
                        best_overall_val_loss = avg_val_loss
                        # Save best model checkpoint
                        save_model(vqgan, f'{log_model_name}_best', 0, H.log_dir)
                        save_model(optimizer_g, f'{log_optimizer_g_name}_best', 0, H.log_dir)
                        save_model(optimizer_d, f'{log_optimizer_d_name}_best', 0, H.log_dir)
                        train_stats = {
                            'losses': loss_arrays,
                            'codebook_usage': codebook_usage,
                            'steps_per_log': H.steps_per_log,
                            'val_losses': val_loss_arrays,
                            'best_val_loss': best_overall_val_loss,
                            'steps_per_validation': H.steps_per_validation,
                        }
                        save_stats(H, train_stats, step)
                        print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
                    else:
                        epochs_without_improvement += 1
                        print(f"No improvement in validation loss for {epochs_without_improvement} validations.")
    
                        if epochs_without_improvement >= H.early_stopping_patience:
                            print(f"Early stopping triggered after {epochs_without_improvement} validations with no improvement.")
                            break
                    print(f"Validation loss: {avg_val_loss:.4f} | Best: {best_overall_val_loss:.4f}")
                            
                vqgan.train()
                vq_loss.train()

        # Save checkpoints
        if step % H.steps_per_checkpoint == 0 or step == H.train_steps:
            save_model(vqgan, log_model_name, step, H.log_dir)
            save_model(optimizer_g, log_optimizer_g_name, step, H.log_dir)
            save_model(optimizer_d, log_optimizer_d_name, step, H.log_dir)

            train_stats = {
                'losses': loss_arrays,
                'codebook_usage': codebook_usage,
                'steps_per_log': H.steps_per_log,
                'val_losses': val_loss_arrays,
                'best_val_loss': best_overall_val_loss,
                'steps_per_validation': H.steps_per_validation,
            }
            save_stats(H, train_stats, step)


if __name__ == "__main__":
    main()