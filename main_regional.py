# main_regional.py
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import numpy as np

from models.multimodal_regional_densenet121 import MultimodalRegionalDenseNet121
from utils.regional_dataset_loader import RegionalCXRDataset
from XAI_models.xai_regional_gradcam import run_regional_gradcam, analyze_tb_regions

def main():
    # ⚙️ Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # 📦 Dataset and transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = RegionalCXRDataset(root_dir="data/", transform=transform, generate_synthetic_regions=True)
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # 🧠 Model setup
    model = MultimodalRegionalDenseNet121(clinical_input_dim=3, num_regions=6).to(device)
    model_path = "regional_model_best.pth"
    
    # 🔁 Load or suggest training
    if os.path.exists(model_path):
        print("📥 Loading trained regional model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print("\n🎯 Model loaded successfully!")
        print("Region mapping:")
        region_names = model.get_region_names()
        for i, name in enumerate(region_names):
            print(f"  {i}: {name}")
        
        # 🔍 Demonstration: Test the model on a few samples
        print("\n🧪 Testing regional predictions...")
        
        # Create a small test loader
        test_size = min(20, len(dataset))
        test_indices = np.random.choice(len(dataset), test_size, replace=False)
        test_samples = [dataset[i] for i in test_indices]
        
        tb_cases = []
        normal_cases = []
        
        for i, (image, clinical, global_label, regional_labels) in enumerate(test_samples):
            if global_label == 1 and len(tb_cases) < 3:
                tb_cases.append(test_indices[i])
            elif global_label == 0 and len(normal_cases) < 2:
                normal_cases.append(test_indices[i])
        
        # 🔍 Run Regional GradCAM on interesting cases
        print("\n🔍 Running Regional GradCAM Analysis...")
        
        # Analyze TB cases
        for i, case_idx in enumerate(tb_cases):
            print(f"\n--- TB Case {i+1} (Sample {case_idx}) ---")
            run_regional_gradcam(model, dataset, device, index=case_idx)
        
        # Analyze normal cases
        for i, case_idx in enumerate(normal_cases):
            print(f"\n--- Normal Case {i+1} (Sample {case_idx}) ---")
            run_regional_gradcam(model, dataset, device, index=case_idx)
        
        # 📊 Regional distribution analysis
        print("\n📊 Analyzing regional TB distribution...")
        analyze_tb_regions(model, dataset, device, num_samples=50)
        
        # 🎯 Interactive demo
        print("\n🎮 Interactive Regional Analysis")
        print("Enter a sample index to analyze (or 'q' to quit):")
        
        while True:
            try:
                user_input = input("Sample index: ").strip()
                if user_input.lower() == 'q':
                    break
                
                idx = int(user_input)
                if 0 <= idx < len(dataset):
                    result = run_regional_gradcam(model, dataset, device, index=idx)
                    
                    # Show prediction summary
                    print(f"\n📋 Summary for Sample {idx}:")
                    print(f"Global: {'TB' if result['global_label'] == 1 else 'Normal'} (Pred: {result['global_prediction'][1]:.3f})")
                    
                    for i, region_name in enumerate(region_names):
                        pred_prob = result['regional_predictions'][i][1].item()
                        true_label = result['regional_labels'][i].item()
                        status = "✓" if (pred_prob > 0.5) == true_label else "✗"
                        print(f"  {region_name}: {pred_prob:.3f} (GT: {true_label}) {status}")
                    
                else:
                    print(f"Invalid index. Please enter a number between 0 and {len(dataset)-1}")
                    
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")
            except KeyboardInterrupt:
                break
    
    else:
        print("❌ Regional model not found!")
        print(f"Please train the model first by running:")
        print(f"  python train_regional_model.py")
        print(f"\nThis will create the regional model with location-specific TB detection capabilities.")
        
        # Show what the regional model would provide
        print("\n🎯 Regional Model Capabilities:")
        print("✅ Global TB classification (TB vs Normal)")
        print("✅ Region-specific TB detection:")
        
        region_names = MultimodalRegionalDenseNet121().get_region_names()
        for i, name in enumerate(region_names):
            print(f"   • {name.replace('_', ' ').title()}")
        
        print("\n✅ Regional GradCAM visualizations")
        print("✅ Location-specific explanations")
        print("✅ Multi-label training with regional supervision")

if __name__ == "__main__":
    main()
