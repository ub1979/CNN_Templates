# ==============================================================
# Importing the libraries
# ==============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

# ==============================================================
# Define the CNN model
# ==============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 18 * 18, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Apply convolutions with ReLU and max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        # Flatten the output
        x = x.view(-1, 64 * 18 * 18)
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==============================================================
# Function to train the model
# ==============================================================
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to the specified device (CPU or GPU)
        data, target = data.to(device), target.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(data)
        # Calculate loss
        loss = F.cross_entropy(output, target)
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        # Print training statistics
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# ==============================================================
# Function to test the model
# ==============================================================
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to the specified device (CPU or GPU)
            data, target = data.to(device), target.to(device)
            # Forward pass
            output = model(data)
            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # Sum up correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate and print test statistics
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# ==============================================================
# Main function to run the image classifier
# ==============================================================
def main():
    # Set the path to your dataset
    data_dir = 'path/to/your/dataset'

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Get the number of classes
    num_classes = len(full_dataset.classes)

    # Create the model
    model = SimpleCNN(num_classes).to(device)

    # Define optimizer
    optimizer = optim.Adam(model.parameters())

    # Train and test the model
    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Print class labels
    print("Class labels:", full_dataset.class_to_idx)

# ==============================================================
# Run the main function
# ==============================================================
if __name__ == "__main__":
    main()