import os

# Download dataset using Python
os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
os.system("unzip tiny-imagenet-200.zip -d tiny-imagenet")


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
