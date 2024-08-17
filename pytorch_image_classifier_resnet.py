# ==============================================================
# Importing the libraries
# ==============================================================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# ==============================================================
# Define the CNN model using a pre-trained ResNet18
# ==============================================================
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes):
        super(AdvancedCNN, self).__init__()
        # Load pre-trained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Replace the last fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


# ==============================================================
# Function to train the model
# ==============================================================
def train(model, device, train_loader, optimizer, epoch, scheduler):
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
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    # Step the scheduler
    scheduler.step()


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
    return accuracy


# ==============================================================
# Main function to run the image classifier
# ==============================================================
def main():
    # Set the path to your dataset
    data_dir = 'path/to/your/dataset'

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Get the number of classes
    num_classes = len(train_dataset.classes)

    # Create the model
    model = AdvancedCNN(num_classes).to(device)

    # Define optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # Train and test the model
    best_accuracy = 0
    for epoch in range(1, 21):  # 20 epochs
        train(model, device, train_loader, optimizer, epoch, scheduler)
        accuracy = test(model, device, test_loader)

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    # Print class labels
    print("Class labels:", train_dataset.class_to_idx)
    print(f"Best test accuracy: {best_accuracy:.2f}%")


# ==============================================================
# Run the main function
# ==============================================================
if __name__ == "__main__":
    main()