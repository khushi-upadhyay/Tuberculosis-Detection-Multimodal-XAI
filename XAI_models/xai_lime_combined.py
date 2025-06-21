import torch
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image, lime_tabular
from skimage.segmentation import mark_boundaries
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from PIL import Image

def run_lime_combined(model, dataset, device, index=0):
    model.eval()

    # 📥 Get sample
    image_tensor, clinical_tensor, label = dataset[index]
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    # === IMAGE EXPLANATION ===
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485], std=[0.229])  # Adjust channels if needed
    ])

    def image_predict(images):
        model.eval()
        images = [transform(Image.fromarray((img * 255).astype(np.uint8))) for img in images]
        images = torch.stack(images).to(device)
        clinical = clinical_tensor.unsqueeze(0).repeat(len(images), 1).to(device)
        with torch.no_grad():
            outputs = model(images, clinical)
        return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image_np, image_predict, top_labels=1, hide_color=0, num_samples=200)
    temp, mask = explanation.get_image_and_mask(label.item(), positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)

    # === TEXT (CLINICAL) EXPLANATION ===
    clinical_data = clinical_tensor.cpu().numpy()
    feature_names = ["Age", "Sex", "Abnormality"]
    class_names = ["Normal", "TB"]

    def tabular_predict(clinical_array):
        model.eval()
        batch_size = 16  # smaller batches to save memory
        preds_list = []

        for i in range(0, len(clinical_array), batch_size):
            batch = torch.tensor(clinical_array[i:i+batch_size], dtype=torch.float32).to(device)
            # Repeat the single image tensor for batch size
            image_batch = image_tensor.unsqueeze(0).repeat(len(batch), 1, 1, 1).to(device)
            with torch.no_grad():
                preds = model(image_batch, batch)
                preds_list.append(torch.nn.functional.softmax(preds, dim=1).cpu().numpy())

        return np.concatenate(preds_list, axis=0)

    tabular_explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array([clinical_data]),
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    # Reduce num_samples to avoid huge memory usage
    tabular_exp = tabular_explainer.explain_instance(clinical_data, tabular_predict, num_features=3, num_samples=100)
    
    print("📋 Clinical Explanation:")
    for feature, weight in tabular_exp.as_list():
        print(f"🔹 {feature}: {weight:.3f}")

    # === PLOT IMAGE EXPLANATION ===
    plt.imshow(img_boundry)
    plt.title("LIME Image Explanation")
    plt.axis('off')
    plt.show()
