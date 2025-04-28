import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
import numpy as np
import scanpy as sc
from tqdm import tqdm
import os
import pickle
import wandb

from gears.utils import loss_fct, uncertainty_loss_fct, print_sys
from gears.inference import evaluate, compute_metrics # Use standard evaluate/compute_metrics

# Assuming PertData class is in gears_utils
# from gears_utils import PertData
from torch.utils.data import Subset
from torch_geometric.data import DataLoader


def subsampler(total_size, sample_size):
    """Randomly samples indices without replacement."""
    if sample_size > total_size:
        print_sys(f"Warning: Sample size ({sample_size}) > total size ({total_size}). Using total size.")
        sample_size = total_size
    indices = torch.randperm(total_size).tolist()
    subsample_indices = indices[:sample_size]
    return subsample_indices

def calculate_fisher_information(gears_wrapper, dataloader, criterion, device):
    """
    Calculates the diagonal Fisher Information Matrix (FIM) using a subset of data.

    Args:
        gears_wrapper: The GEARS wrapper object containing the model.
        dataloader: DataLoader providing the data subset for FIM calculation.
        criterion: The loss function used for training (e.g., loss_fct, uncertainty_loss_fct).
        device: The torch device ('cpu' or 'cuda').

    Returns:
        dict: A dictionary mapping parameter names to their diagonal FIM estimate.
    """
    fim = {}
    original_requires_grad = {}  # Store original states
    model = gears_wrapper.model # Access model via wrapper

    # Initialize FIM dict and enable requires_grad temporarily
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(True) # Ensure grad is enabled for calculation
        fim[name] = torch.zeros_like(param) # Initialize FIM tensor on the correct device

    model.eval() # Set model to evaluation mode for FIM calculation

    num_samples = 0
    # Compute Fisher Information (sum of squared gradients)
    for batch in tqdm(dataloader, desc="Calculating FIM", leave=False):
        batch.to(device)
        model.zero_grad()

        if gears_wrapper.config['uncertainty']:
            pred, logvar = model(batch)
            loss = criterion(pred, logvar, batch.y, batch.pert,
                             reg=gears_wrapper.config['uncertainty_reg'],
                             ctrl=gears_wrapper.ctrl_expression,
                             dict_filter=gears_wrapper.dict_filter,
                             direction_lambda=gears_wrapper.config['direction_lambda'])
        else:
            pred = model(batch)
            loss = criterion(pred, batch.y, batch.pert,
                             ctrl=gears_wrapper.ctrl_expression,
                             dict_filter=gears_wrapper.dict_filter,
                             direction_lambda=gears_wrapper.config['direction_lambda'])

        # Check if loss requires grad, otherwise skip backward()
        if loss.requires_grad:
             loss.backward()
        else:
             print_sys("Warning: Loss does not require grad. Skipping backward pass for FIM.")
             continue # Skip this batch if loss doesn't require grad


        current_batch_size = batch.y.size(0) # Number of graphs in the batch
        num_samples += current_batch_size

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Accumulate squared gradients element-wise
                fim[name] += (param.grad.data ** 2) * current_batch_size
            # else:
            #     print_sys(f"Warning: Grad is None for parameter {name} during FIM calculation.")


    # Average FIM over the number of samples
    if num_samples > 0:
        for name in fim:
            fim[name] /= num_samples
    else:
        print_sys("Warning: No samples processed for FIM calculation.")


    # Restore original requires_grad settings and detach FIM tensors
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad[name])
        if name in fim:
             fim[name] = fim[name].detach().cpu().numpy() # Store FIM as numpy array on CPU


    # Aggregate FIM per parameter (sum across all dimensions) for layer ranking
    param_fim_aggregated = {name: np.sum(value) for name, value in fim.items()}

    return param_fim_aggregated # Return aggregated scalar FIM per parameter

# FUN with Fisher: Improving Generalization of Adapter-Based Cross-lingual Transfer with Scheduled Unfreezing 
# (Liu et al., NAACL 2024)
def unfreeze_layer(model, param_fim, only_nonzero=True, verbose=True):
    """
    Unfreezes the parameters of the next most 'stable' (lowest FIM) layer that is currently frozen.

    Args:
        model: The PyTorch model.
        param_fim (dict): Dictionary mapping parameter names to their aggregated scalar FIM values.
        only_nonzero (bool): If True, only consider parameters/layers with FIM > 0.
        verbose (bool): If True, print messages about unfreezing.

    Returns:
        bool: True if all layers with relevant FIM are already unfrozen, False otherwise.
    """
    # Aggregate FIM by layer (using the first part of the parameter name)
    layer_fim_aggregated = {}
    param_frozen_status = {}

    for name, param in model.named_parameters():
        # Use the top-level module name as the layer identifier
        layer_name = name.split('.')[0]
        param_frozen_status[name] = not param.requires_grad

        if name in param_fim:
            current_fim = param_fim[name]
            # Apply filter: only non-zero FIM and only if the param is currently frozen
            if (not only_nonzero or current_fim > 1e-9) and param_frozen_status[name]:
                 layer_fim_aggregated[layer_name] = layer_fim_aggregated.get(layer_name, 0) + current_fim
        # else:
        #      print_sys(f"Warning: Parameter {name} not found in FIM dictionary during unfreeze.")


    if not layer_fim_aggregated:
        if verbose:
            print_sys("No eligible frozen layers found to unfreeze based on FIM criteria.")
        return True # All relevant layers are already unfrozen or have zero/negative FIM

    # Sort layers by their aggregated FIM (ascending - unfreeze least important first)
    # Filter layers where ALL parameters are already unfrozen is implicitly handled
    # because layer_fim_aggregated only includes layers with at least one frozen parameter with FIM > 0.
    sorted_layers = sorted(layer_fim_aggregated.items(), key=lambda item: item[1])

    # Find the first layer in the sorted list that has frozen parameters
    layer_to_unfreeze = None
    for layer_name, fim_val in sorted_layers:
         # Check if any parameter within this layer is actually frozen
         is_layer_frozen = any(name.startswith(layer_name) and param_frozen_status[name] for name in param_frozen_status)
         if is_layer_frozen:
              layer_to_unfreeze = layer_name
              break # Found the layer to unfreeze

    if layer_to_unfreeze is None:
        if verbose:
            print_sys("All layers considered by FIM are already unfrozen.")
        return True # All candidate layers were already unfrozen


    # Unfreeze all parameters belonging to the selected layer
    unfrozen_count = 0
    if verbose:
        print_sys(f"Unfreezing layer: {layer_to_unfreeze} (Aggregated FIM: {layer_fim_aggregated[layer_to_unfreeze]:.4e})")
    for name, param in model.named_parameters():
        if name.startswith(layer_to_unfreeze):
            if not param.requires_grad:
                 param.requires_grad = True
                 unfrozen_count += 1
                # if verbose:
                #     print(f"  - Unfroze parameter: {name}")

    if unfrozen_count == 0:
         print_sys(f"Warning: Layer {layer_to_unfreeze} was selected but no parameters were actually unfrozen.")

    return False # Indicate that a layer was unfrozen (or attempted)


def finetune(gears_wrapper, epochs=20, lr=1e-5, weight_decay=5e-6, k=75,
             scheduler_step_size=1, scheduler_gamma=0.5, clip_grad=1.0,
             use_wandb=True, wandb_project="ml-aml", wandb_exp_name="finetuned-aml-fisher",
             fim_select_method='param', # 'param' or 'layer'
             fim_subset_fraction=0.1):
    """
    Fine-tunes a pre-trained GEARS model using AML data and FIM-based layer unfreezing.

    Args:
        gears_wrapper: Initialized GEARS wrapper with loaded pre-trained model.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for Adam optimizer.
        k (int): Frequency (in steps) for calculating FIM and potentially unfreezing layers.
                 Set to 0 or None to disable FIM-based unfreezing.
        scheduler_step_size (int): StepLR scheduler step size.
        scheduler_gamma (float): StepLR scheduler gamma (multiplicative factor).
        clip_grad (float): Gradient clipping value. Set to None to disable.
        use_wandb (bool): Whether to log metrics to Weights & Biases.
        wandb_project (str): WandB project name.
        wandb_exp_name (str): WandB experiment name.
        fim_select_method (str): Method to select what to unfreeze ('param' or 'layer').
        fim_subset_fraction (float): Fraction of validation data to use for FIM calculation.

    Returns:
        torch.nn.Module: The best performing model based on validation DE MSE.
    """

    device = gears_wrapper.device
    model = gears_wrapper.model # Get model from wrapper
    config = gears_wrapper.config # Get config from wrapper
    dataloader = gears_wrapper.dataloader # Get dataloaders from wrapper

    # --- WandB Setup ---
    if use_wandb:
        try:
            wandb.init(project=wandb_project, name=wandb_exp_name, config={
                "epochs": epochs, "lr": lr, "weight_decay": weight_decay, "k": k,
                "scheduler_step_size": scheduler_step_size, "scheduler_gamma": scheduler_gamma,
                "clip_grad": clip_grad, "model_config": config,
                "dataset": gears_wrapper.pert_data.dataset_name, # Access dataset name via pert_data
                "split": gears_wrapper.pert_data.split, # Access split via pert_data
                "fim_select_method": fim_select_method,
                "fim_subset_fraction": fim_subset_fraction,
            })
            wandb.watch(model, log='all') # Watch gradients and parameters
        except Exception as e:
            print_sys(f"WandB initialization failed: {e}. Disabling WandB.")
            use_wandb = False
    else:
        print_sys("WandB logging is disabled.")


    # --- Freeze All Layers Initially ---
    print_sys("Freezing all model parameters initially...")
    for name, param in model.named_parameters():
        param.requires_grad = False
        # print(f"  - Frozen: {name}") # Debug print
    all_unfrozen = (k is None or k <= 0) # If k is not set, consider all unfrozen from start

    # --- Prepare Dataloaders ---
    print_sys("Checking Dataloaders...")
    train_loader = dataloader.get('train_loader')
    val_loader = dataloader.get('val_loader')
    test_loader = dataloader.get('test_loader') # May be None

    if not train_loader or not val_loader:
        raise ValueError("Training and validation dataloaders are required.")
    if not test_loader:
         print_sys("Warning: Test dataloader not found. Testing will be skipped.")

    # --- Optimizer and Scheduler ---
    # Filter parameters that require gradients (will be empty initially)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


    # --- Loss Function ---
    criterion = uncertainty_loss_fct if config['uncertainty'] else loss_fct

    # --- Training Loop ---
    min_val_de_mse = np.inf
    best_model_state = deepcopy(model.state_dict()) # Store initial state

    # Calculate total steps for FIM schedule if k is active
    total_steps_for_fim = 0
    if not all_unfrozen:
         total_param_count = sum(1 for _ in model.parameters()) # Rough estimate
         total_steps_for_fim = min(k * total_param_count, epochs * len(train_loader)) # Cap at total training steps
         print_sys(f"FIM unfreezing scheduled every {k} steps, estimated up to step {total_steps_for_fim}.")


    print_sys("Starting Fine-tuning...")
    global_step = 0
    for epoch in range(epochs):
        print_sys(f"--- Starting Epoch {epoch+1}/{epochs} ---")
        model.train() # Set model to training mode

        epoch_loss = 0.0
        steps_in_epoch = len(train_loader)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit='batch')):

            # --- FIM Calculation and Unfreezing ---
            if not all_unfrozen and global_step % k == 0 and global_step < total_steps_for_fim:
                print_sys(f"\nStep {global_step}: Calculating Fisher Information...")
                fim_sample_size = max(1, int(fim_subset_fraction * len(val_loader.dataset)))
                print_sys(f"Using {fim_sample_size} samples from validation set for FIM.")
                fim_indices = subsampler(len(val_loader.dataset), fim_sample_size)
                fim_subset = Subset(val_loader.dataset, fim_indices)
                # Use validation batch size or training batch size for FIM loader
                fim_batch_size = val_loader.batch_size or train_loader.batch_size
                fim_loader = DataLoader(fim_subset, batch_size=fim_batch_size, shuffle=False)

                param_fim_aggregated = calculate_fisher_information(gears_wrapper, fim_loader, criterion, device)

                # Unfreeze based on FIM
                all_unfrozen = unfreeze_layer(model, param_fim_aggregated, verbose=True)

                if not all_unfrozen:
                    # Update optimizer with newly unfrozen parameters
                    print_sys("Updating optimizer with newly unfrozen parameters...")
                    # Recreate optimizer to include newly trainable parameters
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                           lr=scheduler.get_last_lr()[0], # Use current LR from scheduler
                                           weight_decay=weight_decay)
                    # Recreate scheduler with the new optimizer
                    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma,
                                       last_epoch=scheduler.last_epoch) # Maintain scheduler state

                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print_sys(f"Total trainable parameters now: {trainable_params}")
                    if use_wandb: wandb.log({"trainable_parameters": trainable_params}, step=global_step)
                else:
                    print_sys("All layers considered by FIM are now unfrozen.")

                model.train() # Ensure model is back in train mode


            # --- Training Step ---
            batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            if config['uncertainty']:
                pred, logvar = model(batch)
                loss = criterion(pred, logvar, batch.y, batch.pert,
                                 reg=config['uncertainty_reg'],
                                 ctrl=gears_wrapper.ctrl_expression,
                                 dict_filter=gears_wrapper.dict_filter,
                                 direction_lambda=config['direction_lambda'])
            else:
                pred = model(batch)
                loss = criterion(pred, batch.y, batch.pert,
                                 ctrl=gears_wrapper.ctrl_expression,
                                 dict_filter=gears_wrapper.dict_filter,
                                 direction_lambda=config['direction_lambda'])

            # Backward pass and optimization
            if loss.requires_grad: # Check if loss requires grad (might not if no params are trainable)
                 loss.backward()
                 if clip_grad is not None:
                     nn.utils.clip_grad_value_(filter(lambda p: p.requires_grad, model.parameters()), clip_value=clip_grad)
                 optimizer.step()
                 epoch_loss += loss.item()
            # else:
                 # print_sys(f"Step {global_step}: Loss does not require grad. Skipping optimizer step.") # Debug

            # Logging
            if use_wandb:
                wandb.log({'train/step_loss': loss.item(), 'epoch': epoch + (step / steps_in_epoch)}, step=global_step)

            if global_step % 50 == 0: # Log every 50 steps
                print_sys(f"  Step {step+1}/{steps_in_epoch} | Train Loss: {loss.item():.4f}")

            global_step += 1
        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        print_sys(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]


        # --- Validation ---
        print_sys("Running validation...")
        val_res = evaluate(val_loader, model, config['uncertainty'], device) # Pass model directly
        # Handle case where evaluate returns None (e.g., empty loader)
        if val_res is None:
             print_sys("Warning: Validation evaluation returned None. Skipping validation metrics.")
             val_metrics = {'mse_de': np.inf} # Set high MSE to avoid saving this model
        else:
             val_metrics, _ = compute_metrics(val_res)

        print_sys(f"Epoch {epoch+1}: Validation Overall MSE: {val_metrics.get('mse', float('nan')):.4f} | "
                  f"Validation Top 20 DE MSE: {val_metrics.get('mse_de', float('nan')):.4f}")

        # --- Log Metrics (WandB) ---
        if use_wandb:
            log_dict = {
                'train/epoch_loss': avg_epoch_loss,
                'val/mse': val_metrics.get('mse', float('nan')),
                'val/pearson': val_metrics.get('pearson', float('nan')),
                'val/mse_de': val_metrics.get('mse_de', float('nan')),
                'val/pearson_de': val_metrics.get('pearson_de', float('nan')),
                'learning_rate': current_lr,
                'epoch': epoch+1 # Log epoch number
            }
            # Optionally log training metrics (can be expensive)
            # train_res = evaluate(train_loader, model, config['uncertainty'], device)
            # train_metrics, _ = compute_metrics(train_res)
            # log_dict.update({'train/mse': train_metrics['mse'], ...})
            wandb.log(log_dict, step=global_step) # Log against global step


        # --- Model Checkpointing ---
        if val_metrics.get('mse_de', float('inf')) < min_val_de_mse:
             min_val_de_mse = val_metrics['mse_de']
             best_model_state = deepcopy(model.state_dict())
             print_sys(f"** New best model found at Epoch {epoch+1} with Validation DE MSE: {min_val_de_mse:.4f} **")
             if use_wandb:
                 # Save best model checkpoint locally and potentially to WandB
                 best_model_path = os.path.join(wandb.run.dir if wandb.run else ".", "GEARS_fine-tuned_best.pt")
                 torch.save(best_model_state, best_model_path)
                 print_sys(f"Best model state saved to {best_model_path}")
                 # wandb.save(best_model_path, base_path=wandb.run.dir if wandb.run else ".") # Save best model artifact


    # --- End of Training ---
    print_sys("Fine-tuning finished!")
    print_sys(f"Best Validation Top 20 DE MSE achieved: {min_val_de_mse:.4f}")

    # Load best model state
    model.load_state_dict(best_model_state)
    gears_wrapper.model = model # Update wrapper with the best model
    gears_wrapper.best_model = model # Also store in best_model attribute if needed


    # --- Final Testing ---
    if test_loader:
        print_sys("Running final evaluation on test set with best model...")
        test_res = evaluate(test_loader, model, config['uncertainty'], device)
        if test_res is None:
             print_sys("Warning: Test evaluation returned None. Skipping test metrics.")
        else:
            test_metrics, test_pert_res = compute_metrics(test_res)
            print_sys(f"Test Overall MSE: {test_metrics.get('mse', float('nan')):.4f} | Test Top 20 DE MSE: {test_metrics.get('mse_de', float('nan')):.4f}")
            if use_wandb:
                wandb.log({
                    'test/mse': test_metrics.get('mse', float('nan')),
                    'test/pearson': test_metrics.get('pearson', float('nan')),
                    'test/mse_de': test_metrics.get('mse_de', float('nan')),
                    'test/pearson_de': test_metrics.get('pearson_de', float('nan')),
                }) # Log final test metrics

            # --- Deeper Analysis (Optional) ---
            # You can add the deeper_analysis and non_dropout_analysis calls here
            # if needed, ensuring gears_wrapper.adata is correctly populated/passed.
            # Example:
            # if hasattr(gears_wrapper, 'adata') and gears_wrapper.adata is not None:
            #     print_sys("Running deeper analysis...")
            #     out = deeper_analysis(gears_wrapper.adata, test_res)
            #     # ... log results ...
            # else:
            #     print_sys("Skipping deeper analysis as adata is not available in the wrapper.")

    else:
        print_sys("Skipping final test evaluation as no test loader was provided.")


    # --- WandB Finish ---
    if use_wandb and wandb.run:
        # Save the final best model as an artifact if not done already
        # final_model_path = os.path.join(wandb.run.dir, "GEARS_fine-tuned_final.pt")
        # torch.save(best_model_state, final_model_path)
        # wandb.save(final_model_path, base_path=wandb.run.dir)
        wandb.finish()

    return model # Return the best model instance 