import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchvision import models
import torch.nn as nn
# -------------------
# Load trained model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("cloud_classification_model.pth", map_location=device, weights_only=False)
model.eval()


# -------------------
# Set transforms (must match training setup)
# -------------------
val_test_transform = transforms.Compose([
    transforms.Resize(120),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225]),
])
# -------------------
# Load test dataset
# -------------------
test_dataset_full = datasets.ImageFolder(root='/Users/lakitha/mintsData/trainingData/cloudCategorization/swimcat/images', transform=val_test_transform)

# If you saved test indices, load them — otherwise, use the full test set:
# Example: assuming you saved test_indices.npy earlier
# import numpy as np
# test_indices = np.load("test_indices.npy")
# from torch.utils.data import Subset
# test_dataset = Subset(test_dataset_full, test_indices)

# Or use full test dataset if you're confident it's correctly split
test_loader = DataLoader(test_dataset_full, batch_size=32, shuffle=False)

class_names = test_dataset_full.classes

# -------------------
# Run predictions
# -------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -------------------
# Classification report
# -------------------
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

# Optional: Save to file
with open("classification_report.txt", "w") as f:
    f.write(report)

# -------------------
# Confusion matrix
# -------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm,xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
# plt.show()
