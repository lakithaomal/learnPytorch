import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import torch.nn.functional as F  # for softmax

# Define your class names (order must match training)
class_names = ['A-sky', 'B-pattern', 'C-thick-dark', 'D-thick-white', 'E-veil']

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))  # 5 classes
model.load_state_dict(torch.load("cloud_classification_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Define the same transforms as val/test
transform = transforms.Compose([
    transforms.Resize(120),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Load your image
img_path = "testImage.png"
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)

    # Get top 1 prediction
    top_prob, predicted = torch.max(probs, 1)
    predicted_label = class_names[predicted.item()]
    confidence = top_prob.item() * 100
    print(f"Predicted class: {predicted_label} ({confidence:.2f}% confidence)")

    # Get top 3 predictions
    topk_probs, topk_indices = torch.topk(probs, k=3)
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs[0]):
        label = class_names[i]
        percent = prob.item() * 100
        print(f"{label}: {percent:.2f}%")