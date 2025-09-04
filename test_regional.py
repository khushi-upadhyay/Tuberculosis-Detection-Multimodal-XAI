#!/usr/bin/env python
# test_regional.py - Quick test of regional components

from models.multimodal_regional_densenet121 import MultimodalRegionalDenseNet121
from utils.regional_dataset_loader import RegionalCXRDataset
from torchvision import transforms
import torch

def test_regional_components():
    print("🧪 Testing Regional TB Detection Components...")
    
    # Test model creation
    try:
        model = MultimodalRegionalDenseNet121()
        print('✅ Regional model created successfully!')
        print(f'   Regions: {model.get_region_names()}')
        print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    except Exception as e:
        print(f'❌ Model creation failed: {e}')
        return

    # Test dataset loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor()
    ])
    
    try:
        dataset = RegionalCXRDataset(root_dir='data/', transform=transform)
        print(f'✅ Dataset loaded: {len(dataset)} samples')
        
        if len(dataset) == 0:
            print('❌ No samples found in dataset')
            return
        
        # Test a sample
        image, clinical, global_label, regional_labels = dataset[0]
        print(f'✅ Sample loaded successfully!')
        print(f'   Image shape: {image.shape}')
        print(f'   Clinical shape: {clinical.shape}')
        print(f'   Global label: {global_label}')
        print(f'   Regional labels: {regional_labels.tolist()}')
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(image.unsqueeze(0), clinical.unsqueeze(0))
            print(f'✅ Forward pass successful!')
            print(f'   Global output shape: {outputs["global"].shape}')
            print(f'   Number of regional outputs: {len(outputs["regions"])}')
            
            # Show predictions
            global_probs = torch.softmax(outputs['global'], dim=1)[0]
            print(f'   Global prediction: TB={global_probs[1]:.3f}, Normal={global_probs[0]:.3f}')
            
            print('   Regional predictions:')
            for i, region_name in enumerate(model.get_region_names()):
                region_probs = torch.softmax(outputs['regions'][i], dim=1)[0]
                print(f'     {region_name}: TB={region_probs[1]:.3f} (Label: {regional_labels[i]})')
        
        print('\n🎯 Regional Model Summary:')
        print('   • Multi-region TB detection (6 lung regions)')
        print('   • Global + regional classification')
        print('   • Synthetic regional labels generated from image analysis')
        print('   • Ready for region-specific GradCAM visualization')
        
    except Exception as e:
        print(f'❌ Dataset or forward pass failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_regional_components()
