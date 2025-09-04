# utils/regional_dataset_loader.py
import os
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class RegionalCXRDataset(Dataset):
    def __init__(self, root_dir, transform=None, generate_synthetic_regions=True):
        """
        Enhanced dataset loader for regional TB detection
        Args:
            root_dir: base data folder path
            transform: torchvision transforms for images
            generate_synthetic_regions: if True, generates synthetic regional labels
        """
        self.transform = transform
        self.generate_synthetic_regions = generate_synthetic_regions
        
        # Dataset paths
        self.dataset_paths = []
        shenzhen_path = os.path.join(root_dir, 'shenzhen')
        self.dataset_paths.append(shenzhen_path)
        
        # Load samples
        self.samples = []
        for dataset_path in self.dataset_paths:
            self.samples.extend(self._load_dataset(dataset_path))
        
        # Region names for reference
        self.region_names = [
            'upper_left', 'upper_right',
            'middle_left', 'middle_right', 
            'lower_left', 'lower_right'
        ]
    
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
            
            # Parse clinical data
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
            
            # Global label (TB present/absent)
            global_label = 1 if img_file.endswith('_1.png') else 0
            
            # Generate regional labels
            if self.generate_synthetic_regions:
                regional_labels = self._generate_regional_labels(img_path, global_label)
            else:
                # Default: if global TB, randomly assign to 1-3 regions
                regional_labels = self._default_regional_labels(global_label)
            
            samples.append((img_path, clinical_data, global_label, regional_labels))
        
        return samples
    
    def _generate_regional_labels(self, img_path, global_label):
        """
        Generate synthetic regional labels based on image analysis
        This is a placeholder - in practice you'd use actual annotations
        """
        regional_labels = [0] * 6  # 6 regions
        
        if global_label == 1:  # TB present
            # Load and analyze image to determine likely TB regions
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Simple heuristic: analyze image intensity patterns
                    h, w = img.shape
                    
                    # Divide image into 6 regions
                    regions = [
                        img[0:h//3, 0:w//2],        # upper_left
                        img[0:h//3, w//2:w],        # upper_right
                        img[h//3:2*h//3, 0:w//2],   # middle_left
                        img[h//3:2*h//3, w//2:w],   # middle_right
                        img[2*h//3:h, 0:w//2],      # lower_left
                        img[2*h//3:h, w//2:w]       # lower_right
                    ]
                    
                    # Analyze each region (simple intensity variance)
                    variances = [np.var(region) for region in regions]
                    
                    # Assign TB to regions with higher variance (indicating abnormalities)
                    # In TB cases, typically 1-3 regions are affected
                    threshold = np.percentile(variances, 60)  # Top 40% variance regions
                    num_affected = np.random.randint(1, 4)  # 1-3 regions affected
                    
                    # Select top variance regions
                    region_indices = np.argsort(variances)[-num_affected:]
                    for idx in region_indices:
                        regional_labels[idx] = 1
                        
            except Exception as e:
                print(f"Warning: Could not analyze image {img_path}: {e}")
                # Fallback to random assignment
                regional_labels = self._default_regional_labels(global_label)
        
        return regional_labels
    
    def _default_regional_labels(self, global_label):
        """Default regional label assignment"""
        regional_labels = [0] * 6
        
        if global_label == 1:  # TB present
            # Randomly assign to 1-3 regions
            num_affected = np.random.randint(1, 4)
            affected_regions = np.random.choice(6, num_affected, replace=False)
            for region_idx in affected_regions:
                regional_labels[region_idx] = 1
        
        return regional_labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, clinical_data, global_label, regional_labels = self.samples[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Clinical data tensor
        clinical_tensor = torch.tensor([
            clinical_data['age'],
            clinical_data['sex'],
            clinical_data['abnormality']
        ], dtype=torch.float32)
        
        # Labels
        global_label_tensor = torch.tensor(global_label, dtype=torch.long)
        regional_labels_tensor = torch.tensor(regional_labels, dtype=torch.long)
        
        return image, clinical_tensor, global_label_tensor, regional_labels_tensor
