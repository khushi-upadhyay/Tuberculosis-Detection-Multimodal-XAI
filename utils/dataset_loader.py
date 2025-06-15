import os
import re
from PIL import Image
import torch
from torch.utils.data import Dataset

class CXRDataset(Dataset):
    def __init__(self, root_dir, use_montgomery=False, transform=None):
        """
        root_dir: base data folder path (e.g. 'data/')
        use_montgomery: whether to include montgomery dataset along with shenzhen
        transform: torchvision transforms for images
        """
        self.transform = transform

      
        self.dataset_paths = []
        # Shenzhen dataset path
        shenzhen_path = os.path.join(root_dir, 'shenzhen')
        self.dataset_paths.append(shenzhen_path)
        # Montgomery dataset path (optional)
        # if use_montgomery:
        #     montgomery_path = os.path.join(root_dir, 'montgomery')
        #     self.dataset_paths.append(montgomery_path)

        # List to store tuples: (img_path, clinical_data_dict, label)
        self.samples = []
        for dataset_path in self.dataset_paths:
            self.samples.extend(self._load_dataset(dataset_path))

    def _load_dataset(self, base_dir):
        images_dir = os.path.join(base_dir, 'CXR_png')
        clinical_dir = os.path.join(base_dir, 'ClinicalReadings')

        samples = []

        for img_file in os.listdir(images_dir):
            if not img_file.endswith('.png'):
                continue

            img_path = os.path.join(images_dir, img_file)
            clinical_file = img_file.replace('.png', '.txt')
            clinical_path = os.path.join(clinical_dir, clinical_file)

            if not os.path.exists(clinical_path):
                print(f"Warning: clinical file missing for {img_file}, skipping.")
                continue

            with open(clinical_path, 'r') as f:
                lines = f.read().strip().split('\n')

            if len(lines) < 2:
                print(f"Warning: clinical file {clinical_file} incomplete, skipping.")
                continue

            line1 = lines[0].lower()
            tokens = line1.split()
            if len(tokens) < 2:
                print(f"Warning: clinical file {clinical_file} format unexpected, skipping.")
                continue

            sex_str = tokens[0]
            age_str = tokens[1]

            age_match = re.search(r'\d+', age_str)
            if age_match:
                age = float(age_match.group())
            else:
                print(f"Warning: cannot parse age in {clinical_file}, skipping.")
                continue

         
            sex = 0 if sex_str.startswith('m') else 1

            abnormality_str = lines[1].lower()
            abnormality = 0 if 'normal' in abnormality_str else 1

            clinical_data = {
                'age': age,
                'sex': sex,
                'abnormality': abnormality
            }

           
            label = 1 if img_file.endswith('_1.png') else 0

            samples.append((img_path, clinical_data, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, clinical_data, label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        
        clinical_tensor = torch.tensor([
            clinical_data['age'],
            clinical_data['sex'],
            clinical_data['abnormality']
        ], dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, clinical_tensor, label_tensor
