import os
from PIL import Image
from torch.utils.data import Dataset

class CXRDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for folder in folder_paths:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith('.png'):
                        self.image_paths.append(os.path.join(root, file))
                        # Label: 1 if filename ends with '_1.png', else 0
                        label = 1 if file.endswith('_1.png') else 0
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
