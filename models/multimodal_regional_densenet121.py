# models/multimodal_regional_densenet121.py
import torch
import torch.nn as nn
from torchvision.models import densenet121

class MultimodalRegionalDenseNet121(nn.Module):
    def __init__(self, clinical_input_dim=3, num_regions=6):
        """
        Multi-region TB detection model
        Args:
            clinical_input_dim: Number of clinical features (age, sex, abnormality)
            num_regions: Number of lung regions to predict (default: 6)
                        Regions: upper_left, upper_right, middle_left, middle_right, lower_left, lower_right
        """
        super().__init__()
        self.num_regions = num_regions
        
        # Image feature extractor (DenseNet121 backbone)
        self.image_model = densenet121(pretrained=True)
        self.image_model.classifier = nn.Identity()  # Remove final classifier
        
        # Clinical data processor
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        
        # Combined feature processor
        combined_dim = 1024 + 32  # DenseNet features + clinical features
        
        # Region-specific classifiers
        self.region_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)  # Binary: TB present/absent in this region
            ) for _ in range(num_regions)
        ])
        
        # Global TB classifier (overall TB presence)
        self.global_classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Binary: TB present/absent globally
        )
    
    def forward(self, image, clinical):
        # Extract image features
        img_feat = self.image_model(image)  # Shape: [batch_size, 1024]
        
        # Process clinical data
        clin_feat = self.clinical_fc(clinical)  # Shape: [batch_size, 32]
        
        # Combine features
        combined_feat = torch.cat([img_feat, clin_feat], dim=1)  # Shape: [batch_size, 1056]
        
        # Region-specific predictions
        region_outputs = []
        for region_classifier in self.region_classifiers:
            region_output = region_classifier(combined_feat)
            region_outputs.append(region_output)
        
        # Global prediction
        global_output = self.global_classifier(combined_feat)
        
        return {
            'global': global_output,
            'regions': region_outputs,
            'combined_features': combined_feat
        }
    
    def get_region_names(self):
        """Return list of region names for interpretation"""
        return [
            'upper_left', 'upper_right',
            'middle_left', 'middle_right', 
            'lower_left', 'lower_right'
        ]
