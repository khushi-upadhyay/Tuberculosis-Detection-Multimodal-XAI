# main.py

import torch
from models.multimodal_densenet121 import MultimodalDenseNet121
from utils.dataset_loader import CXRDataset
from train_and_eval import train, evaluate
# from XAI_models.xai_gradcam import run_gradcam
from XAI_models.xai_lime import run_lime

from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


model = MultimodalDenseNet121().to(device)
model_path = "best_model.pth"

if os.path.exists(model_path):
    print("📥 Loading trained model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("🔧 Training model...")
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(3):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\n📊 Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

    torch.save(model.state_dict(), model_path)
    print("✅ Model saved to best_model.pth")

# 🔍 Run GradCAM 
# run_gradcam(model, dataset, device,index=8)

# 🔍 Run LIME
run_lime(model, dataset, device, index=8)

