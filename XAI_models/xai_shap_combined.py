import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from PIL import Image
from skimage.segmentation import mark_boundaries

def run_shap_combined(model, dataset, device, index=0):
    model.eval()

    # === Extract Sample ===
    image_tensor, clinical_tensor, label = dataset[index]
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    print("🔍 True Label:", "TB" if label.item() == 1 else "Normal")

    # === Clinical (Tabular) SHAP Explanation ===
    clinical_data = clinical_tensor.cpu().numpy()
    feature_names = ["Age", "Sex", "Abnormality"]
    class_names = ["Normal", "TB"]

    # Model wrapper for SHAP - only for tabular input
    def model_clinical_only(clinical_array):
        model.eval()
        clinical_batch = torch.tensor(clinical_array, dtype=torch.float32).to(device)
        image_batch = image_tensor.unsqueeze(0).repeat(len(clinical_batch), 1, 1, 1).to(device)
        with torch.no_grad():
            preds = model(image_batch, clinical_batch)
        return torch.nn.functional.softmax(preds, dim=1).cpu().numpy()

    # KernelExplainer with small background (can be clinical_data itself)
    explainer = shap.KernelExplainer(model_clinical_only, np.array([clinical_data]))
    shap_values = explainer.shap_values(clinical_data.reshape(1, -1), nsamples=200)

    # Bar plot for SHAP explanation
    print("📊 Clinical SHAP Explanation:")
    shap.summary_plot(shap_values, features=clinical_data.reshape(1, -1), feature_names=feature_names)

    # === Image Display ===
    plt.imshow(image_np)
    plt.title("Original Chest X-ray Image")
    plt.axis('off')
    plt.show()
