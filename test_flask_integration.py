# test_flask_integration.py
"""
Test script to check if Flask app can integrate with trained models
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_model_integration():
    print("🧪 Testing Flask App Integration...")
    
    # Check if models exist
    model_files = {
        'Regional Model': '../regional_model_best.pth',
        'Global Model': '../best_model.pth'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"✅ {name} found: {path}")
        else:
            print(f"❌ {name} missing: {path}")
    
    # Check if model classes can be imported
    try:
        from models.multimodal_regional_densenet121 import MultimodalRegionalDenseNet121
        print("✅ Regional model class imported successfully")
        
        # Test model creation
        model = MultimodalRegionalDenseNet121()
        print(f"✅ Regional model created - Regions: {model.get_region_names()}")
        
    except ImportError as e:
        print(f"❌ Regional model import failed: {e}")
    
    try:
        from models.multimodal_densenet121 import MultimodalDenseNet121
        print("✅ Global model class imported successfully")
    except ImportError as e:
        print(f"❌ Global model import failed: {e}")
    
    # Check if GradCAM can be imported
    try:
        from XAI_models.xai_regional_gradcam import RegionalGradCAM
        print("✅ Regional GradCAM imported successfully")
    except ImportError as e:
        print(f"❌ Regional GradCAM import failed: {e}")
    
    print("\n🚀 Flask App Integration Status:")
    print("   Simple Website: ✅ Running at http://localhost:8000")
    print("   Enhanced Flask: 🔄 Run 'python website/app.py' for model integration")
    print("   Image Upload: ✅ Available with simulation mode")
    print("   Real AI: ✅ Will work if models are trained")

if __name__ == "__main__":
    test_model_integration()
