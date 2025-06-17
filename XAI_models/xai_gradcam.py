from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import normalize

# Setup the target layer
target_layer = model.image_model.features[-1]  # Last conv layer

# Create GradCAM object
cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

# Input sample
image_tensor, clinical_tensor, label = dataset[0]
input_tensor = image_tensor.unsqueeze(0).to(device)
clinical_tensor = clinical_tensor.unsqueeze(0).to(device)

# Forward pass
model.eval()
grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True)[0]  # Get CAM for 1st sample

# Convert for display
rgb_image = image_tensor.permute(1, 2, 0).cpu().numpy()
visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

import matplotlib.pyplot as plt
plt.imshow(visualization)
plt.title("Grad-CAM Heatmap")
plt.axis("off")
plt.show()
