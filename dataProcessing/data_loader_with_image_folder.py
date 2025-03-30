from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dataset_path = "../data/version2/train"
test_dataset_path  = "../data/version2/test"

# Compose concatenates transform operations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load train dataset
train_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Print basic info about train_loader
print("=== Train Loader Info ===")
print("Batch size:", train_loader.batch_size)
print("Shuffle: True")  # Manually specify since DataLoader does not expose this
print("Number of batches per epoch:", len(train_loader))
print("Dataset size:", len(train_loader.dataset))

# Load test dataset
test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Print basic info about test_loader
print("\n=== Test Loader Info ===")
print("Batch size:", test_loader.batch_size)
print("Shuffle: True")  # Same as above
print("Number of batches per epoch:", len(test_loader))
print("Dataset size:", len(test_loader.dataset))
