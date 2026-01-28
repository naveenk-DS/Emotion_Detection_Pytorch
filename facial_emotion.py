# facial_emotion.py
# Real-Time Human Emotion Detection - Training File
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------- CONFIG --------
DATA_DIR = r"F:\naveen\Project_Own\Emotion_Detection\processed_data"   # angry, happy, sad, disgust, neutral, confused
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.0003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- IMAGE PREPROCESSING --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------- LOAD DATASET --------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Emotion Classes:", class_names)
print("Total Images:", len(dataset))

# -------- MODEL (TRANSFER LEARNING - RESNET18) --------
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------- TRAINING --------
print("Training Started...")

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} started")
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# -------- SAVE TRAINED MODEL --------
torch.save({
    "model_state": model.state_dict(),
    "class_names": class_names
}, "model.pth")

print("✅ Training completed. Model saved as model.pth")
