import torch

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    total_samples = 0

    for images, clinical_data, labels in dataloader:
        images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, clinical_data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, clinical_data, labels in dataloader:
            images, clinical_data, labels = images.to(device), clinical_data.to(device), labels.to(device)

            outputs = model(images, clinical_data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples * 100
    return avg_loss, accuracy
