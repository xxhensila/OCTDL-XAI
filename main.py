# Import necessary libraries
import random                  # For seeding random number generator (reproducibility)
import numpy as np             # For numerical operations and seeding
from torch.utils.tensorboard import SummaryWriter  # For logging metrics to TensorBoard

# Import custom utility functions and modules
from utils.func import *       # Loads many utility functions (e.g., save config, print logs)
from train import train, evaluate   # Your training and evaluation functions
from utils.metrics import Estimator # Tracks metrics like accuracy, F1-score, etc.
from data.builder import generate_dataset  # Loads and splits your dataset into train/val/test
from modules.builder import generate_model # Builds the deep learning model based on config

# Hydra is used to handle configuration files cleanly
import hydra
from omegaconf import DictConfig  # Lets us use the config as a structured dictionary

# This is your main entry function. Hydra will load the config file (OCTDL.yaml) automatically.
@hydra.main(version_base=None, config_path="./configs", config_name="OCTDL")
def main(cfg: DictConfig) -> None:
    """
    Main function that manages output folder and calls the training workflow.
    cfg: DictConfig object from Hydra, holds everything defined in OCTDL.yaml
    """

    # Set path where models, logs, etc. will be saved
    save_path = cfg.base.save_path

    # Check if save_path already exists
    if os.path.exists(save_path):
        if cfg.base.overwrite:
            # If overwrite is allowed, warn and continue
            print_msg('Save path {} exists and will be overwrited.'.format(save_path), warning=True)
        else:
            # If not, create a new folder with suffix (_1, _2, ...) to avoid overwriting
            new_save_path = add_path_suffix(save_path)
            cfg.base.save_path = new_save_path
            warning = 'Save path {} exists. New save path is set to be {}.'.format(save_path, new_save_path)
            print_msg(warning, warning=True)

    # Create the save directory if it doesn't exist
    os.makedirs(cfg.base.save_path, exist_ok=True)

    # Start the actual training/evaluation pipeline
    worker(cfg)


def worker(cfg):
    """
    Handles the main model training and evaluation logic.
    """

    # If a specific random seed is set (not -1), use it for reproducibility
    if cfg.base.random_seed != -1:
        seed = cfg.base.random_seed
        set_random_seed(seed, cfg.base.cudnn_deterministic)

    # Set up the TensorBoard logger to log metrics
    log_path = os.path.join(cfg.base.save_path, 'log')
    logger = SummaryWriter(log_path)

    # ----- TRAINING START -----

    # Create the model (e.g., ResNet50) using the config
    model = generate_model(cfg)

    # Load the train, test, and validation datasets
    train_dataset, test_dataset, val_dataset = generate_dataset(cfg)

    # Initialize the metrics tracker (e.g., accuracy, F1, AUC, etc.)
    estimator = Estimator(cfg.train.metrics, cfg.data.num_classes, cfg.train.criterion)

    # Start the training loop
    train(
        cfg=cfg,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        estimator=estimator,
        logger=logger
    )

    # ----- TESTING ON BEST MODEL -----

    print('Performance of the best validation model:')

    # Load the best model weights saved during training
    checkpoint = os.path.join(cfg.base.save_path, 'best_validation_weights.pt')
    cfg.train.checkpoint = checkpoint

    # Re-initialize the model and load the best weights
    model = generate_model(cfg)

    # Evaluate the model on the test dataset
    evaluate(cfg, model, test_dataset, estimator)

    # ----- TESTING ON FINAL MODEL -----

    print('Performance of the final model:')

    # Load the final model (last epoch's weights)
    checkpoint = os.path.join(cfg.base.save_path, 'final_weights.pt')
    cfg.train.checkpoint = checkpoint

    # Re-initialize the model and load the final weights
    model = generate_model(cfg)

    # Evaluate the final model on the test set
    evaluate(cfg, model, test_dataset, estimator)


def set_random_seed(seed, deterministic=False):
    """
    Sets random seed across all relevant libraries to make results reproducible.
    deterministic=True ensures exact repeatability, but may reduce training speed.
    """
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy random
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.backends.cudnn.deterministic = deterministic  # Force deterministic backend for cudnn


# Run the main function if the file is executed directly (not imported)
if __name__ == '__main__':
    main()
