import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

def run_shap(model, dataset, device, index=0):
    model.eval()

    # Get clinical data
    test_image, test_clinical, label = dataset[index]
    test_clinical_np = test_clinical.cpu().numpy()

    # Background: small batch of clinical data
    background_clinical = torch.stack([dataset[i][1] for i in range(10)]).cpu().numpy()

    # Define model that takes only clinical input (image fixed)
    def model_clinical_only(clinical_array):
        clinical_tensor = torch.tensor(clinical_array, dtype=torch.float32).to(device)
        image_tensor = test_image.unsqueeze(0).repeat(len(clinical_tensor), 1, 1, 1).to(device)
        with torch.no_grad():
            outputs = model(image_tensor, clinical_tensor)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    # SHAP KernelExplainer
    explainer = shap.KernelExplainer(model_clinical_only, background_clinical)

    # Compute SHAP values for the clinical sample
    shap_values = explainer.shap_values(test_clinical_np.reshape(1, -1), nsamples=200)

    # Feature names (update if you have more)
    feature_names = ["Age", "Sex", "Abnormality"]

    # Plot summary as bar chart
    print("📋 SHAP Explanation for Clinical Features:")
    shap.summary_plot(shap_values, features=test_clinical_np.reshape(1, -1), feature_names=feature_names)
