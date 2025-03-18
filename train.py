import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        # add concolutional layers to reduce the input dimension while adding more channels
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2) # the first two parameters are the input # of channels and output # of channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=2)
        self.fc1 = nn.Linear(128*56*56, 200)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x= x.flatten(start_dim=1)  # ALWAYS FLATTEN BEFORE APPLYING THE LINEAR LAYER
        x = self.fc1(x)

        return x

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
