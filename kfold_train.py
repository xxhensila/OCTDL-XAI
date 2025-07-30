import os
import yaml
import torch
import pandas as pd
from easydict import EasyDict as edict
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from utils.estimator_helper import get_estimator

from train import train
from modules.builder import generate_model
from data.oct_dataset import OCTDataset  # Custom dataset class
from utils.func import save_weights, print_msg  # Use only valid imports


from torchvision import transforms

# Add this inside your run_kfold() loop, before dataset instantiation:
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Or any fixed size you want
    transforms.ToTensor()
])


# Load YAML config into EasyDict
def load_config(config_path):
    with open(config_path, "r") as f:
        cfg = edict(yaml.safe_load(f))
    return cfg

# Function to run k-fold CV
def run_kfold(cfg_path, n_splits=5):
    cfg = load_config(cfg_path)

    # Load CSV that must include file_name, disease, patient_id
    df = pd.read_csv(cfg.data.csv_path)
    groups = df['patient_id']

    gkf = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['disease'], groups)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        # Split metadata
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_dataset = OCTDataset(train_df, cfg, transform=resize_transform)
        val_dataset = OCTDataset(val_df, cfg, transform=resize_transform)

        # Create Model
        model = generate_model(cfg)

        # Update save path for this fold
        fold_path = os.path.join(cfg.base.save_path, f"fold_{fold}")
        cfg.base.save_path = fold_path
        os.makedirs(fold_path, exist_ok=True)

        # Dummy estimator and logger (adjust as needed)
        estimator = get_estimator(cfg)
        logger = None

        # Train the model
        train(cfg, model, train_dataset, val_dataset, estimator, logger=logger)

if __name__ == "__main__":
    run_kfold("configs/OCTDL.yaml", n_splits=5)
