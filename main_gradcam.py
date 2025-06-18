import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.multimodal_densenet121 import MultimodalDenseNet121
from utils.dataset_loader import CXRDataset
from XAI_models.xai_gradcam import run_gradcam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
dataset = CXRDataset(root_dir="data/", transform=transform)


model = MultimodalDenseNet121().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("✅ Model loaded from best_model.pth")


run_gradcam(model, dataset, device)
