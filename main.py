# # main.py

# import torch
# from models.multimodal_densenet121 import MultimodalDenseNet121
# from utils.dataset_loader import CXRDataset
# from train_and_eval import train, evaluate
# from XAI_models.xai_gradcam import run_gradcam
# # from XAI_models.xai_lime import run_lime
# # from XAI_models.xai_lime_combined import run_lime_combined
# # from XAI_models.xai_shap_combined import run_shap_combined
# from XAI_models.xai_shap import run_shap

# from torch.utils.data import random_split, DataLoader
# from torchvision import transforms
# import os


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# dataset = CXRDataset(root_dir="data/", transform=transform)

# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = random_split(dataset, [train_size, val_size])
# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=32)


# model = MultimodalDenseNet121().to(device)
# model_path = "best_model.pth"

# if os.path.exists(model_path):
#     print("📥 Loading trained model...")
#     model.load_state_dict(torch.load(model_path, map_location=device))
# else:
#     print("🔧 Training model...")
#     import torch.nn as nn
#     import torch.optim as optim
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)

#     for epoch in range(3):
#         train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
#         val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#         print(f"\n📊 Epoch {epoch+1}")
#         print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
#         print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

#     torch.save(model.state_dict(), model_path)
#     print("✅ Model saved to best_model.pth")

# # 🔍 Run GradCAM 
# run_gradcam(model, dataset, device,index=89)

# # 🔍 Run LIME
# # run_lime(model, dataset, device, index=411)

# # indexes_to_check = [3, 9, 89,165, 353, 595,615,435]

# # for idx in indexes_to_check:
# #     print(f"\n🔍 Running SHAP for sample index: {idx}")
#     # run_lime_combined(model, dataset, device, index=idx)
#     # run_shap_combined(model, dataset, device, index=idx)
#     # run_shap(model, dataset, device, index=idx)
#     # run_gradcam(model, dataset, device,index=8)



import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

from models.multimodal_densenet121 import MultimodalDenseNet121
from utils.dataset_loader import CXRDataset
from train_and_eval import train, evaluate
from XAI_models.xai_gradcam import run_gradcam
# from XAI_models.xai_lime import run_lime
# from XAI_models.xai_lime_combined import run_lime_combined
# from XAI_models.xai_shap import run_shap

# ⚙️ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📦 Dataset and transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CXRDataset(root_dir="data/", transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# 🧠 Model
model = MultimodalDenseNet121().to(device)
model_path = "best_model.pth"

# 🔁 Load or train model
if os.path.exists(model_path):
    print("📥 Loading trained model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("🔧 Training model for 5 epochs...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\n📊 Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

        # 💾 Save model only at Epoch 4 (i.e., epoch index 3)
        if epoch == 3:
            torch.save(model.state_dict(), model_path)
            print(f"✅ Model from Epoch {epoch + 1} saved to best_model.pth")

# 🔍 Run GradCAM
run_gradcam(model, dataset, device, index=89)
