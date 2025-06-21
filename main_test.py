# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from models.multimodal_densenet121 import MultimodalDenseNet121
# from utils.dataset_loader import CXRDataset
# from train_and_eval import evaluate
# import torch.nn as nn


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# full_dataset = CXRDataset(root_dir="data/", transform=transform)

# # Split- train (80%), val (10%), test (10%)
# total_size = len(full_dataset)
# train_size = int(0.8 * total_size)
# val_size = int(0.1 * total_size)
# test_size = total_size - train_size - val_size
# train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

# test_loader = DataLoader(test_ds, batch_size=32)


# model = MultimodalDenseNet121().to(device)
# model.load_state_dict(torch.load("best_model.pth", map_location=device))
# model.eval()

# criterion = nn.CrossEntropyLoss()


# test_loss, test_acc = evaluate(model, test_loader, criterion, device)

# print(f"\n Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%\n")


import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models.multimodal_densenet121 import MultimodalDenseNet121
from utils.dataset_loader import CXRDataset
from train_and_eval import evaluate

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = CXRDataset(root_dir="data/", transform=transform)
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_ds, batch_size=32)

# === Model ===
model = MultimodalDenseNet121().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# === Evaluate Loss/Accuracy ===
criterion = nn.CrossEntropyLoss()
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"\n🧪 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")


# === Get Predictions ===
y_true, y_pred = [], []
with torch.no_grad():
    for images, clinical_data, labels in test_loader:
        images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)
        outputs = model(images, clinical_data)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# === Metrics ===
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["Normal", "TB"])

print("\n📊 Classification Report:\n", report)

# === Save Directory ===
os.makedirs("results", exist_ok=True)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "TB"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")
plt.close()

# === Bar Plot for Precision, Recall, F1 ===
metrics = {"Precision": precision, "Recall": recall, "F1-score": f1}
plt.figure(figsize=(6, 4))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="Set2")
plt.ylim(0, 1.1)
plt.title("Evaluation Metrics")
plt.ylabel("Score")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.savefig("results/evaluation_metrics.png")
plt.close()

# === Save classification report to .txt ===
with open("results/classification_report.txt", "w") as f:
    f.write("Test Accuracy: {:.2f}%\n".format(test_acc))
    f.write("Test Loss: {:.4f}\n\n".format(test_loss))
    f.write(report)

print("✅ All results saved to 'results/' directory.")
