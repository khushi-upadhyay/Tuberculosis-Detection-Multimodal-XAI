#!/usr/bin/env python3
"""
Flask web application for Regional TB Detection
This creates a full web interface that can integrate with your trained models
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.multimodal_regional_densenet121 import MultimodalRegionalDenseNet121
    from models.multimodal_densenet121 import MultimodalDenseNet121
    from XAI_models.xai_regional_gradcam import RegionalGradCAM
    from utils.regional_dataset_loader import RegionalCXRDataset
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Models not available: {e}")
    MODEL_AVAILABLE = False

app = Flask(__name__)

# Global variables for models
regional_model = None
global_model = None
device = None

def load_models():
    """Load the trained models"""
    global regional_model, global_model, device
    
    if not MODEL_AVAILABLE:
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load regional model
        regional_model = MultimodalRegionalDenseNet121().to(device)
        if os.path.exists("../regional_model_best.pth"):
            regional_model.load_state_dict(torch.load("../regional_model_best.pth", map_location=device))
            regional_model.eval()
            print("✅ Regional model loaded")
        else:
            print("⚠️  Regional model not found")
            
        # Load global model (if available)
        global_model = MultimodalDenseNet121().to(device)
        if os.path.exists("../best_model.pth"):
            global_model.load_state_dict(torch.load("../best_model.pth", map_location=device))
            global_model.eval()
            print("✅ Global model loaded")
        else:
            print("⚠️  Global model not found")
            
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for analyzing uploaded images"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract image data (base64 encoded)
        image_data = data.get('image', '')
        age = data.get('age', 45)
        gender = data.get('gender', 0)
        abnormality = data.get('abnormality', 0)
        
        if not image_data:
            return jsonify({'error': 'No image data provided'})
        
        # Parse base64 image
        try:
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'})
        
        # Perform analysis
        if MODEL_AVAILABLE and regional_model is not None:
            result = analyze_with_real_model(image, age, gender, abnormality)
        else:
            result = analyze_with_simulation(image, age, gender, abnormality)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_used': 'real' if (MODEL_AVAILABLE and regional_model is not None) else 'simulation'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def analyze_with_real_model(image, age, gender, abnormality):
    """Analyze image using the actual trained regional model"""
    try:
        from torchvision import transforms
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Prepare clinical data
        clinical_tensor = torch.tensor([[age, gender, abnormality]], dtype=torch.float32).to(device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = regional_model(image_tensor, clinical_tensor)
            
            # Global prediction
            global_probs = torch.softmax(outputs['global'], dim=1)[0]
            global_tb_prob = global_probs[1].item()
            
            # Regional predictions
            regional_predictions = []
            region_names = regional_model.get_region_names()
            
            for i, region_name in enumerate(region_names):
                region_probs = torch.softmax(outputs['regions'][i], dim=1)[0]
                region_tb_prob = region_probs[1].item()
                
                regional_predictions.append({
                    'region': region_name.replace('_', ' ').title(),
                    'prediction': region_tb_prob,
                    'label': 1 if region_tb_prob > 0.5 else 0
                })
        
        # Generate GradCAM if available
        try:
            from XAI_models.xai_regional_gradcam import RegionalGradCAM
            gradcam = RegionalGradCAM(regional_model)
            
            # Generate global heatmap
            global_heatmap = gradcam.generate_gradcam(image_tensor, clinical_tensor, target_region=None)
            
            # Generate regional heatmaps
            regional_heatmaps = []
            for i in range(len(region_names)):
                regional_heatmap = gradcam.generate_gradcam(image_tensor, clinical_tensor, target_region=i)
                regional_heatmaps.append(regional_heatmap.tolist())  # Convert to list for JSON
                
            gradcam_available = True
        except Exception as e:
            print(f"GradCAM generation failed: {e}")
            gradcam_available = False
            regional_heatmaps = []
        
        # Determine overall diagnosis
        is_tb = global_tb_prob > 0.5
        confidence = 'High' if global_tb_prob > 0.7 or global_tb_prob < 0.3 else 'Medium'
        
        # Generate description
        description = generate_real_analysis_description(is_tb, regional_predictions)
        
        return {
            'type': 'Real AI Analysis',
            'global': {
                'normal': 1 - global_tb_prob,
                'tb': global_tb_prob
            },
            'global_label': 'TB' if is_tb else 'Normal',
            'regional': regional_predictions,
            'description': description,
            'clinical_data': {
                'age': age,
                'gender': 'Male' if gender == 0 else 'Female',
                'abnormality': 'Yes' if abnormality == 1 else 'No'
            },
            'confidence': confidence,
            'gradcam_available': gradcam_available,
            'regional_heatmaps': regional_heatmaps
        }
        
    except Exception as e:
        print(f"Real model analysis failed: {e}")
        # Fallback to simulation
        return analyze_with_simulation(image, age, gender, abnormality)

def analyze_with_simulation(image, age, gender, abnormality):
    """Simulate analysis for demo purposes"""
    import random
    
    # Simulate some analysis based on image characteristics
    # In a real scenario, you might do some basic image processing here
    
    # Generate realistic TB probability based on age and other factors
    base_tb_prob = 0.1  # Base probability
    
    # Age factor (higher risk for older patients)
    if age > 60:
        base_tb_prob += 0.3
    elif age > 40:
        base_tb_prob += 0.2
    
    # Previous abnormality factor
    if abnormality == 1:
        base_tb_prob += 0.4
    
    # Add some randomness
    tb_probability = base_tb_prob + random.uniform(-0.2, 0.3)
    tb_probability = max(0.05, min(0.95, tb_probability))  # Clamp between 5% and 95%
    
    is_tb = tb_probability > 0.5
    
    # Generate regional predictions
    regional_predictions = []
    region_names = ['Upper Left', 'Upper Right', 'Middle Left', 'Middle Right', 'Lower Left', 'Lower Right']
    
    for region_name in region_names:
        if is_tb:
            # If TB, some regions more likely affected
            region_prob = random.uniform(0.2, 0.9) if random.random() > 0.4 else random.uniform(0.05, 0.4)
        else:
            # If normal, all regions should have low probability
            region_prob = random.uniform(0.05, 0.3)
        
        regional_predictions.append({
            'region': region_name,
            'prediction': region_prob,
            'label': 1 if region_prob > 0.5 else 0
        })
    
    confidence = 'High' if tb_probability > 0.7 or tb_probability < 0.3 else 'Medium'
    description = generate_real_analysis_description(is_tb, regional_predictions)
    
    return {
        'type': 'Simulated AI Analysis',
        'global': {
            'normal': 1 - tb_probability,
            'tb': tb_probability
        },
        'global_label': 'TB' if is_tb else 'Normal',
        'regional': regional_predictions,
        'description': description,
        'clinical_data': {
            'age': age,
            'gender': 'Male' if gender == 0 else 'Female',
            'abnormality': 'Yes' if abnormality == 1 else 'No'
        },
        'confidence': confidence,
        'gradcam_available': False,
        'regional_heatmaps': []
    }

def generate_real_analysis_description(is_tb, regional_predictions):
    """Generate analysis description based on predictions"""
    if not is_tb:
        return 'AI analysis indicates no significant tuberculosis features detected. All lung regions show patterns consistent with normal chest radiography.'
    
    affected_regions = [r['region'] for r in regional_predictions if r['prediction'] > 0.5]
    
    if len(affected_regions) == 0:
        return 'AI analysis suggests possible tuberculosis with low regional confidence. Additional clinical correlation recommended.'
    elif len(affected_regions) == 1:
        return f'AI analysis indicates tuberculosis features primarily detected in the {affected_regions[0]} region. Localized disease pattern observed.'
    elif len(affected_regions) <= 3:
        return f'AI analysis shows multi-regional tuberculosis involvement in: {", ".join(affected_regions)}. Moderate disease extent suggested.'
    else:
        return f'AI analysis indicates extensive tuberculosis involvement across multiple regions: {", ".join(affected_regions)}. Widespread disease pattern detected.'

def get_demo_predictions(sample_id=1):
    """Get demo predictions for visualization"""
    demo_data = {
        1: {
            'type': 'TB Case 1',
            'global': {'normal': 0.108, 'tb': 0.892},
            'global_label': 'TB',
            'regional': [
                {'region': 'Upper Left', 'prediction': 0.834, 'label': 1},
                {'region': 'Upper Right', 'prediction': 0.123, 'label': 0},
                {'region': 'Middle Left', 'prediction': 0.756, 'label': 1},
                {'region': 'Middle Right', 'prediction': 0.089, 'label': 0},
                {'region': 'Lower Left', 'prediction': 0.234, 'label': 0},
                {'region': 'Lower Right', 'prediction': 0.691, 'label': 1}
            ],
            'description': 'Multi-focal TB with involvement in upper left, middle left, and lower right regions.'
        },
        2: {
            'type': 'TB Case 2',
            'global': {'normal': 0.245, 'tb': 0.755},
            'global_label': 'TB',
            'regional': [
                {'region': 'Upper Left', 'prediction': 0.567, 'label': 1},
                {'region': 'Upper Right', 'prediction': 0.689, 'label': 1},
                {'region': 'Middle Left', 'prediction': 0.234, 'label': 0},
                {'region': 'Middle Right', 'prediction': 0.156, 'label': 0},
                {'region': 'Lower Left', 'prediction': 0.123, 'label': 0},
                {'region': 'Lower Right', 'prediction': 0.098, 'label': 0}
            ],
            'description': 'Bilateral upper lobe TB, common presentation pattern.'
        },
        3: {
            'type': 'Normal Case',
            'global': {'normal': 0.912, 'tb': 0.088},
            'global_label': 'Normal',
            'regional': [
                {'region': 'Upper Left', 'prediction': 0.067, 'label': 0},
                {'region': 'Upper Right', 'prediction': 0.134, 'label': 0},
                {'region': 'Middle Left', 'prediction': 0.089, 'label': 0},
                {'region': 'Middle Right', 'prediction': 0.076, 'label': 0},
                {'region': 'Lower Left', 'prediction': 0.098, 'label': 0},
                {'region': 'Lower Right', 'prediction': 0.112, 'label': 0}
            ],
            'description': 'Healthy chest X-ray with no signs of tuberculosis.'
        }
    }
    
    return demo_data.get(sample_id, demo_data[1])

@app.route('/api/status')
def status():
    """API endpoint to check model status"""
    return jsonify({
        'models_available': MODEL_AVAILABLE,
        'regional_model_loaded': regional_model is not None,
        'global_model_loaded': global_model is not None,
        'device': str(device) if device else 'N/A'
    })

if __name__ == '__main__':
    print("🚀 Starting Regional TB Detection Web Application")
    print("=" * 60)
    
    # Try to load models
    models_loaded = load_models()
    
    if models_loaded:
        print("✅ Models loaded successfully")
    else:
        print("⚠️  Running in demo mode (models not available)")
    
    print(f"🌐 Web application will start at: http://localhost:5000")
    print(f"🛑 Press Ctrl+C to stop the server")
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
