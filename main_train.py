# # main_train.py

# from models.multimodal_resnet50 import MultimodalResNet50  
# from utils.dataset_loader import CXRDataset
# from torch.utils.data import random_split, DataLoader
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from train_and_eval import train, evaluate

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset and transform
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

# # Model
# model = MultimodalResNet50().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Train
# for epoch in range(3):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#     print(f"\n📊 Epoch {epoch+1}")
#     print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
#     print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}\n")


# main_train.py

# # from models.multimodal_efficientnet import MultimodalEfficientNetB0 
# from models.multimodal_densenet121 import MultimodalDenseNet121

# from utils.dataset_loader import CXRDataset
# from torch.utils.data import random_split, DataLoader
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from train_and_eval import train, evaluate

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset and transform
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

# # Model
# model = MultimodalDenseNet121().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Train for 3 epochs
# for epoch in range(5):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#     print(f"\n📊 Epoch {epoch+1}")
#     print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
#     print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")




# # from models.multimodal_efficientnet import MultimodalEfficientNetB0 
# from models.multimodal_densenet121 import MultimodalDenseNet121

# from utils.dataset_loader import CXRDataset
# from torch.utils.data import random_split, DataLoader
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from train_and_eval import train, evaluate

# # Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Dataset and transform
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

# # Model
# model = MultimodalDenseNet121().to(device)

# # Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Train for 5 epochs
# for epoch in range(5):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
#     val_loss, val_acc = evaluate(model, val_loader, criterion, device)

#     print(f"\n📊 Epoch {epoch+1}")
#     print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
#     print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

#     # ✅ Save only at Epoch 4 (index = 3)
#     if epoch == 3:
#         torch.save(model.state_dict(), "best_model.pth")
#         print(f"✅ Saved model at Epoch 4 with Val Accuracy: {val_acc:.2f}%")

# print("\n🎉 Training complete.")



# main_train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
import numpy as np

from models.multimodal_densenet121 import MultimodalDenseNet121
from utils.dataset_loader import CXRDataset
from train_and_eval import train, evaluate

# ==== Set random seed for reproducibility ====
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==== Setup ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==== Load Dataset ====
dataset = CXRDataset(root_dir="data/", transform=transform)

# ==== Split Dataset ====
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# ==== Model ====
model = MultimodalDenseNet121().to(device)

# ==== Loss and Optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ==== Print Hyperparameters ====
print("🚀 Hyperparameters used:")
print(f"Model: MultimodalDenseNet121")
print(f"Optimizer: Adam")
print(f"Learning Rate: 1e-4")
print(f"Loss Function: CrossEntropyLoss")
print(f"Batch Size: 32")
print(f"Epochs: 5")
print(f"Random Seed: {SEED}")
print("=" * 50)

# ==== Training ====
for epoch in range(5):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"\n📊 Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

    # Commented: Don't overwrite best_model.pth
    # if epoch == 3:
    #     torch.save(model.state_dict(), "best_model.pth")
    #     print(f"✅ Saved model at Epoch 4 with Val Accuracy: {val_acc:.2f}%")

print("\n🎉 Training complete.")
