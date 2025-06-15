import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils.dataset_loader import CXRDataset

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = CXRDataset([
        "data/shenzhen",
        "data/montgomery"
    ], transform=transform)

    print(f"Total dataset size: {len(dataset)}")  # Debug print

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    images, labels = next(iter(train_loader))
    print(f"Loaded batch of shape: {images.shape}, Labels: {labels}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    main()
