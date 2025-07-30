import timm  # PyTorch image models library for pre-trained models
import torch  # PyTorch library for tensors and deep learning
from utils.func import print_msg, select_out_features  # Import utility functions

# Function to generate the model, load weights, and send to device
def generate_model(cfg):
    model = build_model(cfg)  # Build the model based on config

    if cfg.train.checkpoint:  # If a checkpoint path is specified
        weights = torch.load(cfg.train.checkpoint)  # Load weights from checkpoint
        model.load_state_dict(weights, strict=True)  # Load weights into model
        print_msg('Load weights form {}'.format(cfg.train.checkpoint))  # Print confirmation message

    model = model.to(cfg.base.device)  # Move model to specified device (CPU/GPU)

    return model  # Return the prepared model

# Function to build the model based on config settings
def build_model(cfg):
    network = cfg.train.network  # Get network name from config
    out_features = select_out_features(  # Determine number of output features
        cfg.data.num_classes,  # Number of classes
        cfg.train.criterion  # Training criterion
    )

    if 'vit' in network or 'swin' in network:  # Check if model is Vision Transformer or Swin
        model = timm.create_model(  # Create model using timm with image size
            network,  # Model architecture name
            img_size=cfg.data.input_size,  # Input image size
            in_chans=cfg.data.in_channels,  # Number of input channels
            num_classes=out_features,  # Number of output classes
            pretrained=cfg.train.pretrained,  # Whether to use pretrained weights
        )
    else:  # For other model types that don’t use img_size
        model = timm.create_model(  # Create model without img_size parameter
            network,  # Model architecture name
            in_chans=cfg.data.in_channels,  # Number of input channels
            num_classes=out_features,  # Number of output classes
            pretrained=cfg.train.pretrained,  # Whether to use pretrained weights
        )

    return model  # Return the built model
