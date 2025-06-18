# XAI_models/xai_lime.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from PIL import Image

def run_lime(model, dataset, device, index=0):
    model.eval()

    
    image_tensor, clinical_tensor, label = dataset[index]
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

 
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229])  
    ])

    def batch_predict(images):
        model.eval()
        images = [transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images]
        images = torch.stack(images).to(device)
        clinical = clinical_tensor.unsqueeze(0).repeat(len(images), 1).to(device)
        with torch.no_grad():
            outputs = model(images, clinical)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=2, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    plt.imshow(img_boundry)
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.show()
