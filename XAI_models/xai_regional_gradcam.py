# XAI_models/xai_regional_gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os

class RegionalGradCAM:
    def __init__(self, model, target_layers=None):
        """
        Regional GradCAM for multi-region TB detection
        Args:
            model: trained regional model
            target_layers: list of layer names to use for GradCAM
        """
        self.model = model
        self.model.eval()
        
        # Default target layer (last conv layer of DenseNet)
        if target_layers is None:
            self.target_layers = ['image_model.features.norm5']
        else:
            self.target_layers = target_layers
        
        self.gradients = {}
        self.activations = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0]
            return hook
        
        # Register hooks for target layers
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def generate_gradcam(self, input_image, clinical_data, target_region=None, target_class=1):
        """
        Generate GradCAM heatmap
        Args:
            input_image: input tensor [1, 3, H, W]
            clinical_data: clinical tensor [1, clinical_dim]
            target_region: specific region index (0-5) or None for global
            target_class: class to generate CAM for (default: 1 for TB)
        """
        # Forward pass
        output = self.model(input_image, clinical_data)
        
        # Get target output
        if target_region is not None:
            target_output = output['regions'][target_region]
        else:
            target_output = output['global']
        
        # Get the score for target class
        class_score = target_output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)
        
        # Get gradients and activations
        target_layer = self.target_layers[0]
        gradients = self.gradients[target_layer]
        activations = self.activations[target_layer]
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute GradCAM
        gradcam = torch.sum(weights * activations, dim=1, keepdim=True)
        gradcam = F.relu(gradcam)
        
        # Normalize
        gradcam = gradcam.squeeze().cpu().numpy()
        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)
        
        return gradcam
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = alpha * heatmap_colored + (1 - alpha) * image
        
        return overlayed.astype(np.uint8)

def run_regional_gradcam(model, dataset, device, index=0, save_dir="results/regional_gradcam"):
    """
    Run Regional GradCAM analysis on a specific sample
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get sample
    image, clinical, global_label, regional_labels = dataset[index]
    image_tensor = image.unsqueeze(0).to(device)
    clinical_tensor = clinical.unsqueeze(0).to(device)
    
    # Convert image for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    
    # Initialize GradCAM
    gradcam = RegionalGradCAM(model)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(image_tensor, clinical_tensor)
        global_pred = torch.softmax(outputs['global'], dim=1)[0]
        regional_preds = [torch.softmax(region_out, dim=1)[0] for region_out in outputs['regions']]
    
    # Region names
    region_names = [
        'Upper Left', 'Upper Right',
        'Middle Left', 'Middle Right', 
        'Lower Left', 'Lower Right'
    ]
    
    print(f"\n🔍 Regional GradCAM Analysis - Sample {index}")
    print(f"Global Label: {'TB' if global_label == 1 else 'Normal'}")
    print(f"Global Prediction: TB={global_pred[1]:.3f}, Normal={global_pred[0]:.3f}")
    print(f"Regional Labels: {regional_labels.tolist()}")
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # Global GradCAM
    global_heatmap = gradcam.generate_gradcam(image_tensor, clinical_tensor, target_region=None)
    global_overlay = gradcam.overlay_heatmap(image_np, global_heatmap)
    axes[0, 1].imshow(global_overlay)
    axes[0, 1].set_title(f"Global GradCAM\nPred: {global_pred[1]:.3f}")
    axes[0, 1].axis('off')
    
    # Regional predictions text
    axes[0, 2].axis('off')
    pred_text = "Regional Predictions:\n\n"
    for i, (name, pred) in enumerate(zip(region_names, regional_preds)):
        tb_prob = pred[1].item()
        label = regional_labels[i].item()
        pred_text += f"{name}: {tb_prob:.3f} (Label: {label})\n"
    axes[0, 2].text(0.1, 0.5, pred_text, fontsize=10, verticalalignment='center')
    
    # Regional GradCAMs
    for i in range(6):
        row = 1 + i // 3
        col = i % 3
        
        # Generate regional heatmap
        regional_heatmap = gradcam.generate_gradcam(
            image_tensor, clinical_tensor, target_region=i, target_class=1
        )
        regional_overlay = gradcam.overlay_heatmap(image_np, regional_heatmap)
        
        axes[row, col].imshow(regional_overlay)
        
        # Title with prediction and ground truth
        tb_prob = regional_preds[i][1].item()
        label = regional_labels[i].item()
        color = 'red' if label == 1 else 'green'
        axes[row, col].set_title(
            f"{region_names[i]}\nPred: {tb_prob:.3f} | GT: {label}",
            color=color, fontweight='bold'
        )
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/regional_gradcam_sample_{index}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual heatmaps
    individual_dir = os.path.join(save_dir, f"sample_{index}_individual")
    os.makedirs(individual_dir, exist_ok=True)
    
    # Save original
    cv2.imwrite(f"{individual_dir}/original.png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    # Save global heatmap
    cv2.imwrite(f"{individual_dir}/global_heatmap.png", (global_heatmap * 255).astype(np.uint8))
    cv2.imwrite(f"{individual_dir}/global_overlay.png", cv2.cvtColor(global_overlay, cv2.COLOR_RGB2BGR))
    
    # Save regional heatmaps
    for i, region_name in enumerate(region_names):
        regional_heatmap = gradcam.generate_gradcam(
            image_tensor, clinical_tensor, target_region=i, target_class=1
        )
        regional_overlay = gradcam.overlay_heatmap(image_np, regional_heatmap)
        
        region_file = region_name.lower().replace(' ', '_')
        cv2.imwrite(f"{individual_dir}/{region_file}_heatmap.png", (regional_heatmap * 255).astype(np.uint8))
        cv2.imwrite(f"{individual_dir}/{region_file}_overlay.png", cv2.cvtColor(regional_overlay, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Regional GradCAM visualizations saved to {save_dir}")
    
    return {
        'global_prediction': global_pred,
        'regional_predictions': regional_preds,
        'global_label': global_label,
        'regional_labels': regional_labels
    }

def analyze_tb_regions(model, dataset, device, num_samples=10):
    """
    Analyze TB distribution across regions for multiple samples
    """
    region_names = [
        'Upper Left', 'Upper Right',
        'Middle Left', 'Middle Right', 
        'Lower Left', 'Lower Right'
    ]
    
    regional_predictions = {name: [] for name in region_names}
    regional_labels = {name: [] for name in region_names}
    
    print(f"\n🔬 Analyzing TB distribution across {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        image, clinical, global_label, reg_labels = dataset[i]
        
        if global_label == 1:  # Only analyze TB cases
            image_tensor = image.unsqueeze(0).to(device)
            clinical_tensor = clinical.unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor, clinical_tensor)
                reg_preds = [torch.softmax(region_out, dim=1)[0] for region_out in outputs['regions']]
            
            for j, region_name in enumerate(region_names):
                regional_predictions[region_name].append(reg_preds[j][1].item())
                regional_labels[region_name].append(reg_labels[j].item())
    
    # Summary statistics
    print("\n📊 Regional TB Analysis Summary:")
    print("Region          | Avg Pred | Label Freq | Samples")
    print("-" * 50)
    
    for region_name in region_names:
        if regional_predictions[region_name]:
            avg_pred = np.mean(regional_predictions[region_name])
            label_freq = np.mean(regional_labels[region_name])
            n_samples = len(regional_predictions[region_name])
            print(f"{region_name:14} | {avg_pred:8.3f} | {label_freq:10.3f} | {n_samples:7d}")
    
    return regional_predictions, regional_labels
