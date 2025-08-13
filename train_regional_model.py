# train_regional_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models.multimodal_regional_densenet121 import MultimodalRegionalDenseNet121
from utils.regional_dataset_loader import RegionalCXRDataset

def regional_train_epoch(model, train_loader, optimizer, criterion_global, criterion_regional, device, alpha=0.5):
    """
    Train one epoch with both global and regional losses
    Args:
        alpha: weight for combining global and regional losses (0.5 = equal weight)
    """
    model.train()
    total_loss = 0
    global_correct = 0
    regional_correct = 0
    total_samples = 0
    total_regional_predictions = 0
    
    for images, clinical, global_labels, regional_labels in train_loader:
        images = images.to(device)
        clinical = clinical.to(device)
        global_labels = global_labels.to(device)
        regional_labels = regional_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, clinical)
        
        # Global loss
        global_loss = criterion_global(outputs['global'], global_labels)
        
        # Regional losses
        regional_losses = []
        for i, region_output in enumerate(outputs['regions']):
            region_loss = criterion_regional(region_output, regional_labels[:, i])
            regional_losses.append(region_loss)
        
        regional_loss = torch.stack(regional_losses).mean()
        
        # Combined loss
        total_loss_batch = alpha * global_loss + (1 - alpha) * regional_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # Statistics
        total_loss += total_loss_batch.item()
        
        # Global accuracy
        _, global_pred = torch.max(outputs['global'], 1)
        global_correct += (global_pred == global_labels).sum().item()
        
        # Regional accuracy
        for i, region_output in enumerate(outputs['regions']):
            _, region_pred = torch.max(region_output, 1)
            regional_correct += (region_pred == regional_labels[:, i]).sum().item()
            total_regional_predictions += regional_labels.size(0)
        
        total_samples += global_labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    global_acc = 100 * global_correct / total_samples
    regional_acc = 100 * regional_correct / total_regional_predictions
    
    return avg_loss, global_acc, regional_acc

def regional_evaluate(model, val_loader, criterion_global, criterion_regional, device, alpha=0.5):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    global_correct = 0
    regional_correct = 0
    total_samples = 0
    total_regional_predictions = 0
    
    # For detailed analysis
    all_global_true = []
    all_global_pred = []
    all_regional_true = []
    all_regional_pred = []
    
    with torch.no_grad():
        for images, clinical, global_labels, regional_labels in val_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            global_labels = global_labels.to(device)
            regional_labels = regional_labels.to(device)
            
            # Forward pass
            outputs = model(images, clinical)
            
            # Global loss
            global_loss = criterion_global(outputs['global'], global_labels)
            
            # Regional losses
            regional_losses = []
            for i, region_output in enumerate(outputs['regions']):
                region_loss = criterion_regional(region_output, regional_labels[:, i])
                regional_losses.append(region_loss)
            
            regional_loss = torch.stack(regional_losses).mean()
            total_loss_batch = alpha * global_loss + (1 - alpha) * regional_loss
            total_loss += total_loss_batch.item()
            
            # Global predictions
            _, global_pred = torch.max(outputs['global'], 1)
            global_correct += (global_pred == global_labels).sum().item()
            all_global_true.extend(global_labels.cpu().numpy())
            all_global_pred.extend(global_pred.cpu().numpy())
            
            # Regional predictions
            batch_regional_pred = []
            for i, region_output in enumerate(outputs['regions']):
                _, region_pred = torch.max(region_output, 1)
                regional_correct += (region_pred == regional_labels[:, i]).sum().item()
                batch_regional_pred.append(region_pred.cpu().numpy())
                total_regional_predictions += regional_labels.size(0)
            
            # Store regional predictions (transpose to get [batch, regions])
            batch_regional_pred = np.array(batch_regional_pred).T
            all_regional_pred.extend(batch_regional_pred)
            all_regional_true.extend(regional_labels.cpu().numpy())
            
            total_samples += global_labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    global_acc = 100 * global_correct / total_samples
    regional_acc = 100 * regional_correct / total_regional_predictions
    
    return avg_loss, global_acc, regional_acc, all_global_true, all_global_pred, all_regional_true, all_regional_pred

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load regional dataset
    dataset = RegionalCXRDataset(root_dir="data/", transform=transform, generate_synthetic_regions=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Region names: {dataset.region_names}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    
    # Model
    model = MultimodalRegionalDenseNet121(clinical_input_dim=3, num_regions=6).to(device)
    
    # Loss functions
    criterion_global = nn.CrossEntropyLoss()
    criterion_regional = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    alpha = 0.6  # Weight for global vs regional loss
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print(f"Loss combination: α={alpha} (global) + {1-alpha} (regional)\n")
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_global_acc, train_regional_acc = regional_train_epoch(
            model, train_loader, optimizer, criterion_global, criterion_regional, device, alpha
        )
        
        # Validation
        val_loss, val_global_acc, val_regional_acc, _, _, _, _ = regional_evaluate(
            model, val_loader, criterion_global, criterion_regional, device, alpha
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"📊 Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Global Acc: {train_global_acc:.2f}% | Regional Acc: {train_regional_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Global Acc: {val_global_acc:.2f}% | Regional Acc: {val_regional_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "regional_model_best.pth")
            print(f"✅ New best model saved! Val Loss: {val_loss:.4f}\n")
    
    print("🎉 Training completed!")
    
    # Final evaluation with detailed metrics
    print("\n📊 Final Evaluation:")
    model.load_state_dict(torch.load("regional_model_best.pth"))
    val_loss, val_global_acc, val_regional_acc, global_true, global_pred, regional_true, regional_pred = regional_evaluate(
        model, val_loader, criterion_global, criterion_regional, device, alpha
    )
    
    print(f"Global Classification Accuracy: {val_global_acc:.2f}%")
    print(f"Regional Classification Accuracy: {val_regional_acc:.2f}%")
    
    # Global classification report
    print("\n🌍 Global TB Classification Report:")
    print(classification_report(global_true, global_pred, target_names=["Normal", "TB"]))
    
    # Regional analysis
    print("\n🎯 Regional TB Analysis:")
    regional_true = np.array(regional_true)
    regional_pred = np.array(regional_pred)
    
    region_names = dataset.region_names
    for i, region_name in enumerate(region_names):
        region_acc = 100 * np.mean(regional_pred[:, i] == regional_true[:, i])
        tb_cases = np.sum(regional_true[:, i])
        print(f"{region_name:12}: {region_acc:5.1f}% accuracy | {tb_cases:3d} TB cases")

if __name__ == "__main__":
    main()
