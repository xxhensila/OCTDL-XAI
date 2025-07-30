import os  # OS module for file and terminal operations
import sys  # System-specific parameters and functions

import yaml  # YAML parser and writer
import torch  # PyTorch library
import shutil  # File operations like copy
import argparse  # Command-line parsing
import torch.nn as nn  # Neural network components

from tqdm import tqdm  # Progress bar utility
from operator import getitem  # Get item by key from nested structure
from functools import reduce  # Function composition helper
from torch.utils.data import DataLoader  # Load data in batches

from utils.const import regression_loss  # List of regression loss identifiers

# Parse command-line arguments to get config file path
def parse_config():
    parser = argparse.ArgumentParser(allow_abbrev=True)  # Create argument parser
    parser.add_argument(  # Add config path argument
        '-config',
        type=str,
        default='./configs/default.yaml',
        help='Path to the config file.'
    )
    parser.add_argument(  # Optionally print config details
        '-print_config',
        action='store_true',
        default=False,
        help='Print details of configs.'
    )
    args = parser.parse_args()  # Parse arguments
    return args  # Return parsed arguments

# Load configuration file from YAML
def load_config(path):
    with open(path, 'r') as file:  # Open YAML file
        cfg = yaml.load(file, Loader=yaml.FullLoader)  # Load as dictionary
    return cfg  # Return config

# Copy config file to destination directory
def copy_config(src, dst):
    if os.path.split(src)[0] != dst:  # Only copy if not already in destination
        shutil.copy(src, dst)  # Perform copy

# Save configuration dictionary to YAML file
def save_config(config, path):
    with open(path, 'w') as file:  # Open destination file
        yaml.safe_dump(config, file)  # Dump config to file safely

# Compute mean and std of training set images
def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(  # Create dataloader for train set
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.  # Initialize sample counter
    channel_mean = torch.Tensor([0., 0., 0.])  # Initialize mean
    channel_std = torch.Tensor([0., 0., 0.])  # Initialize std
    for samples in tqdm(loader):  # First pass to compute mean
        X, _ = samples  # Get image tensor
        channel_mean += X.mean((2, 3)).sum(0)  # Sum per-channel mean
        num_samples += X.size(0)  # Count samples
    channel_mean /= num_samples  # Compute overall mean

    for samples in tqdm(loader):  # Second pass to compute std
        X, _ = samples  # Get image tensor
        batch_samples = X.size(0)  # Batch size
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)  # Flatten for std computation
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples  # Sum squared diffs
    channel_std = torch.sqrt(channel_std / num_samples)  # Final std

    mean, std = channel_mean.tolist(), channel_std.tolist()  # Convert to lists
    print('mean: {}'.format(mean))  # Print mean
    print('std: {}'.format(std))  # Print std
    return mean, std  # Return mean and std

# Save model weights to disk
def save_weights(model, save_path):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):  # If wrapped
        state_dict = model.module.state_dict()  # Extract inner model
    else:
        state_dict = model.state_dict()  # Get regular state_dict
    torch.save(state_dict, save_path)  # Save to file

# Print formatted message with optional highlighting
def print_msg(msg, appendixs=[], warning=False):
    color = '\033[93m'  # ANSI yellow for warnings
    end = '\033[0m'  # Reset color
    print_fn = (lambda x: print(color + x + end)) if warning else print  # Choose print function

    max_len = len(max([msg, *appendixs], key=len))  # Longest line
    max_len = min(max_len, get_terminal_col())  # Clip to terminal width
    print_fn('=' * max_len)  # Top border
    print_fn(msg)  # Main message
    for appendix in appendixs:  # Print each appendix line
        print_fn(appendix)
    print_fn('=' * max_len)  # Bottom border

# Print all sections of a config dictionary
def print_config(configs):
    for name, config in configs.items():  # For each section
        print('====={}====='.format(name))  # Section title
        _print_config(config)  # Recursively print config
        print('=' * (len(name) + 10))  # Separator
        print()  # Blank line

# Recursive helper to print config content
def _print_config(config, indentation=''):
    for key, value in config.items():  # Loop through keys
        if isinstance(value, dict):  # Nested config
            print('{}{}:'.format(indentation, key))  # Print key
            _print_config(value, indentation + '    ')  # Recurse deeper
        else:
            print('{}{}: {}'.format(indentation, key, value))  # Print key-value pair

# Print dataset summary: size and class count
def print_dataset_info(datasets):
    train_dataset, test_dataset, val_dataset = datasets  # Unpack datasets
    print('=========================')  # Separator
    print('Dataset Loaded.')  # Header
    print('Categories:\t{}'.format(len(train_dataset.classes)))  # Number of classes
    print('Training:\t{}'.format(len(train_dataset)))  # Train size
    print('Validation:\t{}'.format(len(val_dataset)))  # Val size
    print('Test:\t\t{}'.format(len(test_dataset)))  # Test size
    print('=========================')  # Separator

# Inverse normalize a tensor image for display
def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):  # Un-normalize each channel
        t.mul_(s).add_(m)
    return tensor  # Return unnormalized tensor

# Convert labels to one-hot encoding
def one_hot(labels, num_classes, device, dtype):
    y = torch.eye(num_classes, device=device, dtype=dtype)  # Identity matrix
    return y[labels]  # Index rows using labels

# Adjust label tensor type based on loss function
def select_target_type(y, criterion):
    if criterion in ['cross_entropy', 'kappa_loss']:  # Classification
        y = y.long()
    elif criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:  # Regression
        y = y.float()
    elif criterion in ['focal_loss']:  # Special loss
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')  # Error on unknown loss
    return y  # Return converted labels

# Determine output dimension based on loss function
def select_out_features(num_classes, criterion):
    out_features = num_classes  # Default to classification output
    if criterion in regression_loss:  # If regression
        out_features = 1  # Single continuous output
    return out_features  # Return output dimension

# Exit script and print error
def exit_with_error(msg):
    print(msg)  # Print error
    sys.exit(1)  # Exit program

# Update values in nested config dictionary
def config_update(cfg, params):
    keys = get_all_keys(cfg)  # Get all nested keys
    name2key = {key[-1]: key for key in keys}  # Map key names to paths
    names = list(name2key.keys())  # Extract name list
    for key, value in params.items():  # Loop through updates
        if key not in names:  # Invalid key
            raise KeyError('Invalid key: {}'.format(key))
        if names.count(key) > 1:  # Ambiguous key
            raise KeyError('Key {} appears more than once, can not be updated'.format(key))
        ks = name2key[key]  # Get key path
        get_by_path(cfg, ks[:-1])[ks[-1]] = value  # Set value in nested dict

# Retrieve value from nested dictionary using path
def get_by_path(d, path):
    return reduce(getitem, path, d)  # Navigate nested dict

# Get list of all nested keys in config
def get_all_keys(cfg):
    keys = []  # List of key paths
    for key, value in cfg.items():  # Loop through top level
        if isinstance(value, dict):  # Recurse if nested
            keys += [[key] + subkey for subkey in get_all_keys(value)]  # Extend key path
        else:
            keys.append([key])  # Add single key path
    return keys  # Return all paths

# Get number of terminal columns (for formatting)
def get_terminal_col():
    try:
        return os.get_terminal_size().columns  # Return width
    except OSError:
        return 80  # Fallback default

# Add suffix to path if it already exists
def add_path_suffix(path):
    suffix = 0  # Start from 0
    new_path = path  # Copy original path
    while os.path.exists(new_path):  # While path exists
        suffix += 1  # Increment suffix
        new_path = path + '_{}'.format(suffix)  # Append suffix
    return new_path  # Return available path

from torch.utils.tensorboard import SummaryWriter

def get_logger(cfg, fold):
    log_dir = os.path.join(cfg.base.save_path, f"fold_{fold}", "log")
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)

