# from torch.utils.data import DataLoader
# from utils.dataset_loader import CXRDataset
# from torchvision import transforms

# def main():
#     # Define transforms
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#     ])

#     # Create dataset
#     dataset = CXRDataset(root_dir='data/', use_montgomery=True, transform=transform)

#     # Create DataLoader for batching
#     dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

#     num_epochs = 1

#     # Example training loop snippet
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         for images, clinical_data, labels in dataloader:
#             # Your training code here
#             pass

# if __name__ == '__main__':
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from utils.dataset_loader import CXRDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    dataset = CXRDataset(root_dir='data/', use_montgomery=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)  # smaller batch for light load

    # Load pretrained ResNet18 and modify final layer for binary classification
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: TB and Normal
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 1

    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        total = 0
        correct = 0

        for images, clinical_data, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100

        print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

if __name__ == '__main__':
    main()

