import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# TensorBoard setup
writer = SummaryWriter("runs/cloud_classification")

# 1. Image transforms
train_transform = transforms.Compose([
    transforms.Resize(120),
    transforms.CenterCrop(100),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(120),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 2. Load dataset
root_path = '/Users/lakitha/mintsData/trainingData/cloudCategorization/swimcat/images'
dataset_full = datasets.ImageFolder(root=root_path, transform=train_transform)
dataset_val_test = datasets.ImageFolder(root=root_path, transform=val_test_transform)

# 3. Train/val/test split
indices = list(range(len(dataset_full)))
np.random.shuffle(indices)

train_ratio, val_ratio = 0.7, 0.15
train_end = int(train_ratio * len(indices))
val_end = train_end + int(val_ratio * len(indices))

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# Save indices for later evaluation
np.save("train_indices.npy", train_indices)
np.save("val_indices.npy", val_indices)
np.save("test_indices.npy", test_indices)

train_dataset = Subset(dataset_full, train_indices)
val_dataset = Subset(dataset_val_test, val_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 4. Model setup
model = models.resnet18(pretrained=True)
for name, param in model.named_parameters():
    param.requires_grad = "fc" in name

model.fc = nn.Linear(model.fc.in_features, 5)  # Adjust for 5 classes

# Compute class weights
from collections import Counter
label_counts = Counter([dataset_full.targets[i] for i in train_indices])
total_count = sum(label_counts.values())
class_weights = [total_count / label_counts[i] for i in range(len(label_counts))]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Training
num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}\n" + "-" * 20)
    for phase, dataloader in [('train', train_loader), ('val', val_loader)]:
        model.train() if phase == 'train' else model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
        writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

# 6. Save model
torch.save(model.state_dict(), "cloud_classification_model.pth")
print("\n✅ Training complete! Model saved.")
writer.close()
