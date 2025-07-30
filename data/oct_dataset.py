import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class OCTDataset(Dataset):
    def __init__(self, df, cfg, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = cfg.data.image_root  # e.g. "OCTDL_dataset/"
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(cfg.data.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Adjust path according to your folder structure
        img_path = os.path.join(self.root, row['disease'], row['file_name'] + '.jpg')
        image = Image.open(img_path).convert('RGB')

        label = self.label_map[row['disease']]

        if self.transform:
            image = self.transform(image)
        else:
            # Fallback transform to ensure tensor output
            from torchvision import transforms
            image = transforms.ToTensor()(image)

        return image, label
