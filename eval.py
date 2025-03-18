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
        # Test loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy
model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)


print(f'Best validation accuracy: {best_acc:.2f}%')
