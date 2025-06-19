# XAI_models/xai_shap.py

import shap
import torch
import matplotlib.pyplot as plt

def run_shap(model, dataset, device, index=0):
    model.eval()
    background = [dataset[i][0] for i in range(10)]  # small background set
    background = torch.stack(background).to(device)
    background_clinical = torch.stack([dataset[i][1] for i in range(10)]).to(device)

    test_image, test_clinical, label = dataset[index]
    test_image = test_image.unsqueeze(0).to(device)
    test_clinical = test_clinical.unsqueeze(0).to(device)

    def model_forward(x):
        return model(x, test_clinical)

    explainer = shap.GradientExplainer(model_forward, background)
    shap_values = explainer.shap_values(test_image)

    shap.image_plot(shap_values, test_image.cpu().numpy())
