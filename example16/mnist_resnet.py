import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import numpy as np

class TransferLearningNet(nn.Module):
    def __init__(self):
        super(TransferLearningNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Modify the input layer to accept grayscale images (1 channel)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the output layer to fit MNIST classes (10 classes)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x):
        return self.resnet(x)

def train_local(model, optimizer, train_loader, criterion, epochs, device):
    model.train()
    model.to(device)
    losses = []
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses

def evaluate(model, test_loader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

# Function to create non-IID data distribution for each device
def create_non_iid_datasets(train_dataset, num_devices, num_classes_per_device):
    class_labels = train_dataset.targets.unique().tolist()
    num_classes = len(class_labels)
    class_labels_per_device = np.array_split(class_labels, num_devices)
    non_iid_datasets = []
    for i in range(num_devices):
        if num_classes_per_device > num_classes:
            selected_classes = class_labels
        else:
            selected_classes = np.random.choice(class_labels, num_classes_per_device, replace=False)
            # Ensure selected classes are present in the dataset
            selected_classes = [c for c in selected_classes if c in class_labels]
            if len(selected_classes) == 0:
                raise ValueError("No samples found for the selected classes. Please adjust num_classes_per_device or choose different classes.")
        print(f"Device {i}: Selected Classes: {selected_classes}")
        indices = [idx for idx in range(len(train_dataset)) if train_dataset.targets[idx] in selected_classes]
        print(f"Device {i}: Number of Samples: {len(indices)}")
        if len(indices) > 0:
            subset_dataset = torch.utils.data.Subset(train_dataset, indices)
            non_iid_datasets.append(subset_dataset)
    return non_iid_datasets

# Load MNIST data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size of ResNet
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Example usage
num_devices = 10
num_classes_per_device = 5  # Adjust as needed
non_iid_datasets = create_non_iid_datasets(train_dataset, num_devices, num_classes_per_device)

# Print non-empty datasets
for i, dataset in enumerate(non_iid_datasets):
    print(f"Device {i}: Number of Samples: {len(dataset)}")

# Hyperparameters
epochs = 20
batch_size = 32
learning_rate = 0.01

# Initialize global model and optimizer
global_model = TransferLearningNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(global_model.parameters(), lr=learning_rate)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Federated learning loop
for epoch in range(epochs):
    local_losses = []
    local_accuracies = []
    for i, dataset in enumerate(non_iid_datasets):
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        local_model = TransferLearningNet().to(device)
        local_model.load_state_dict(global_model.state_dict())
        local_optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
        losses = train_local(local_model, local_optimizer, train_loader, criterion, epochs, device)
        local_losses.append(losses)
        accuracy = evaluate(local_model, test_dataset, device)
        local_accuracies.append(accuracy)
        print(f"Device {i}: Loss: {np.mean(losses):.4f}, Accuracy: {accuracy:.2f}")
    for global_param, local_params in zip(global_model.parameters(), zip(*[model.parameters() for model in local_model])):
        global_param.data = torch.mean(torch.stack([local_param.data for local_param in local_params]), dim=0)
    global_accuracy = evaluate(global_model, test_dataset, device)
    print(f"Epoch {epoch+1}, Global Accuracy: {global_accuracy:.2f}")

