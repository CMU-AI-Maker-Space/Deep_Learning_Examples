import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 10
num_classes = 100  # At the variant level, this is the number of classes

# Data transforms for FGVCAircraft (resize and normalize as per pre-trained model requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for pre-trained models
])

# Load FGVCAircraft dataset
train_dataset = datasets.FGVCAircraft(root="data", split="train", annotation_level="variant", download=True, transform=transform)
test_dataset = datasets.FGVCAircraft(root="data", split="test", annotation_level="variant", download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the model - using ResNet18 and fine-tuning the final layer for our specific task
model = models.resnet18(weights="IMAGENET1K_V1")
# We basically will change only the last layer of the program to match the number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to("cuda")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to('cuda'), targets.to('cuda')  # Move data to GPU if available

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testing loop
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data, targets = data.to('cuda'), targets.to('cuda')
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Visualize some predictions
model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to('cuda'), targets.to('cuda')
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)

        # Loop through the batch
        for i in range(data.size(0)):
            image = data[i].cpu().permute(1, 2, 0)  # Move to CPU and reshape for display
            actual_label = targets[i].item()
            predicted_label = predicted[i].item()

            # Display the image with matplotlib
            plt.imshow(image)
            plt.title(f'Actual: {actual_label}, Predicted: {predicted_label}')
            plt.axis('off')
            plt.show()

            # Add a short delay before moving to the next image
            time.sleep(1)
            plt.close()
