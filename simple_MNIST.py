import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# Hyperparameters
input_size = 784  # 28x28 images
hidden_size = 128   # Arbitrary. Feel free to play around with this
output_size = 10  # 10 classes for digits 0-9

# The guides don't really cover what these are, but feel free to play around. Can you find out online what these mean?
learning_rate = 0.001 # Arbitrary. Feel free to play around with this
batch_size = 64 # Arbitrary. Feel free to play around with this
num_epochs = 5 # Arbitrary. Feel free to play around with this

# Load MNIST dataset
# Normalizing the images is generally a good idea. It gives a more "stable" set of data
# You also need to transform these images to tensors, which is the data structure pytorch understands
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# We download a training set and a test set directly from sources
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# Dataloaders are wrappers around datasets that shuffle and batch the data together
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # We use this data to train the weights
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # We use this data to test how good our model is


# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        # Convert the input to the hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Add a non-linear function
        self.relu = nn.ReLU()
        # Resize to get the same dimension as the number of labels
        self.fc2 = nn.Linear(hidden_size, output_size)

    # You always need to define a forward function in your network
    # Don't worry abot backprop, pytorch takes care of it for you
    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the image
        # Apply the layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Initialize the network, criterion and optimizer
model = NeuralNet(input_size, hidden_size, output_size)
# Choose a loss - this will say how far from the label our loss is
criterion = nn.CrossEntropyLoss()
# The optimizer will update the weights based on the loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass - weight update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}') # You should see the loss going down

# Testing loop
model.eval() # This switched model from training mode to testing mode
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item() # Count the number of times when the model got the right label

    print(f'Test Accuracy: {100 * correct / total:.2f}%') # This tells how much of the model was right

# This will show the images on the screen to the user, so that they can see how good the model is
model.eval()
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)

        # Loop through the batch
        for i in range(data.size(0)):
            image = data[i].view(28, 28)  # Reshape to display as 28x28 image
            actual_label = targets[i].item()
            predicted_label = predicted[i].item()

            # Display the image with matplotlib
            plt.imshow(image, cmap='gray')
            plt.title(f'Actual: {actual_label}, Predicted: {predicted_label}')
            plt.axis('off')
            plt.show()

            # Add a short delay before moving to the next image
            time.sleep(1)  # Adjust the delay (in seconds) as needed
            plt.close()  # Close the figure to display the next image