import torch
import torch.nn as nn 
import torch.optim as optim 
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np

# 1. Define your image transformations (resize, normalize, etc.)
train_transform = transforms.Compose([
    transforms.Resize(120),                  # Resize shorter side to 120, keep aspect ratio
    transforms.CenterCrop(100),             # Crop the center 100x100 patch
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize(120),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])
# 2. Load the full dataset
train_dataset_full    = datasets.ImageFolder(root='/Users/lakitha/mintsData/trainingData/cloudCategorization/swimcat/images', transform=train_transform)
val_test_dataset_full = datasets.ImageFolder(root='/Users/lakitha/mintsData/trainingData/cloudCategorization/swimcat/images', transform=val_test_transform)

# 3. Split indices manually
total_size = len(train_dataset_full)
indices = list(range(total_size))
np.random.shuffle(indices)

train_ratio = 0.7
val_ratio = 0.15

train_end = int(train_ratio * total_size)
val_end = train_end + int(val_ratio * total_size)

train_indices = indices[:train_end]
val_indices = indices[train_end:val_end]
test_indices = indices[val_end:]

# 4. Subset datasets using correct transforms
train_dataset = Subset(train_dataset_full, train_indices)
val_dataset   = Subset(val_test_dataset_full, val_indices)
test_dataset  = Subset(val_test_dataset_full, test_indices)


print(f"Total samples: {total_size}")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
class_names = train_dataset.dataset.classes
print(class_names)

# 5. Wrap in DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)
test_loader  = DataLoader(test_dataset, batch_size=4)

# 6 Trianing the classifier 
model = models.resnet18(pretrained = True)
for name , param in model.named_parameters():
    if "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9 )

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
num_epochs = 100
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 20)

    for phase, dataloader in [('train', train_loader), ('val', val_loader)]:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

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
        # 👇 Append metrics for plotting
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_acc.item())
print("✅ Training complete!")


# Save the model
torch.save(model, 'cloud_classification_model.pth')


import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Acc')
plt.plot(epochs, val_accuracies, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()

# 🔽 Save the figure
plt.savefig("training_curves.png", dpi=300)  # You can change the filename/format if you like

# 👀 Still show it interactively
plt.show()
