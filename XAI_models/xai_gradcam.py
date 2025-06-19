# ✅ XAI_models/xai_gradcam.py

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

class ImageOnlyWrapper(torch.nn.Module):
    def __init__(self, multimodal_model, dummy_clinical_tensor):
        super().__init__()
        self.model = multimodal_model
        self.dummy_clinical_tensor = dummy_clinical_tensor

    def forward(self, x):
        batch_size = x.shape[0]
        clinical = self.dummy_clinical_tensor.repeat(batch_size, 1).to(x.device)
        return self.model(x, clinical)


def run_gradcam(model, dataset, device, index=0):
    model.eval()

    # 🖼️ Sample input
    image_tensor, clinical_tensor, label = dataset[index]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # 🔧 Wrap model to bypass clinical input for Grad-CAM
    wrapper_model = ImageOnlyWrapper(model, clinical_tensor.unsqueeze(0)).to(device)

    # 🎯 Target the last CNN layer
    target_layer = wrapper_model.model.image_model.features[-1]

    # 🔍 GradCAM
    cam = GradCAM(model=wrapper_model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    # 🎨 Visualize
    rgb_image = image_tensor.permute(1, 2, 0).cpu().numpy()
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")
    plt.show()
