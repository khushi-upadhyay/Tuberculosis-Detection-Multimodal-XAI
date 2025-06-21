import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from PIL import Image

def run_lime(model, dataset, device, index=0):
    model.eval()

    # Get image and clinical features
    image_tensor, clinical_tensor, label_tensor = dataset[index]
    label = label_tensor.item()
    image_np = image_tensor.permute(1, 2, 0).numpy()

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def batch_predict(images):
        model.eval()
        images = [transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images]
        images = torch.stack(images).to(device)
        clinical = clinical_tensor.unsqueeze(0).repeat(len(images), 1).to(device)
        with torch.no_grad():
            outputs = model(images, clinical)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    # LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, batch_predict, top_labels=2, hide_color=0, num_samples=1000)

    print(f"✅ LIME labels: {list(explanation.local_exp.keys())} | True Label: {label}")

    # Use predicted label if actual not in explanation
    if label not in explanation.local_exp:
        label = list(explanation.local_exp.keys())[0]
        print(f"⚠️ Label {label_tensor.item()} not found in explanation. Using predicted label: {label}")

    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    plt.imshow(img_boundry)
    plt.title("LIME Explanation")
    plt.axis('off')
    plt.show()
