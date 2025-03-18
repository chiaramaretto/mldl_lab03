import torch
from torch import nn

import shutil

# Ensure validation annotations exist
val_ann_path = 'tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt'
if not os.path.exists(val_ann_path):
    raise FileNotFoundError(f"{val_ann_path} not found. Check your dataset extraction.")

# Organize validation set
with open(val_ann_path) as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)
        shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', 
                        f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

# Remove original validation images folder
shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')

# Data transformations
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

transform = T.Compose([
    T.Resize((224, 224)),  # Resize images to 224x224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

# Verify dataset sizes
print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

# Create data loaders
train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)
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

