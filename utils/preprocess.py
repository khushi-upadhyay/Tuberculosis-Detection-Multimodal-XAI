# utils/preprocess.py

import torch
from tqdm import tqdm
from torchvision import transforms
from utils.dataset_loader import CXRDataset  
def save_preprocessed_data(root_dir, output_path='processed_data.pt', use_montgomery=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CXRDataset(root_dir=root_dir, use_montgomery=use_montgomery, transform=transform)

    all_images = []
    all_clinical = []
    all_labels = []

    for img, clinical, label in tqdm(dataset, desc="Preprocessing"):
        all_images.append(img)
        all_clinical.append(clinical)
        all_labels.append(label)

    data_dict = {
        'images': torch.stack(all_images),
        'clinical': torch.stack(all_clinical),
        'labels': torch.stack(all_labels)
    }

    torch.save(data_dict, output_path)
    print(f"✅ Saved preprocessed dataset to {output_path}")
