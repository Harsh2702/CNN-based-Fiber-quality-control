# run.py
import torch
from Data_fetch import get_azure_dataset_dynamic,AugmentedDataset
from models.tinyvit import TinyViTBinaryClassifier
from Train import train_model
from Evaluate import evaluate
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = get_azure_dataset_dynamic()
#dataset = torch.utils.data.Subset(dataset, range(300, 600)) #testing

print(f"Total samples: {len(dataset)}")
all_labels = [label.item() for _, label in dataset]
all_labels = np.array(all_labels)


train_size = int(0.7*len(dataset))
val_size   = int(0.15*len(dataset))
test_size  = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

augment_transform = T.Compose([
    T.RandomHorizontalFlip(p=1),
    T.RandomVerticalFlip(p=1),
    T.RandomRotation(5),
    T.ColorJitter(
        brightness=0.2,
        contrast=0.2      
    ),
])

train_ds = AugmentedDataset(train_ds, transform=augment_transform)
val_ds   = AugmentedDataset(val_ds, transform=augment_transform)
test_ds  = AugmentedDataset(test_ds, transform=augment_transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)

# --- Model ---
model = TinyViTBinaryClassifier().to(device) 

# Count number of samples per class
num_io = (all_labels == 0).sum()
num_pseudo = (all_labels == 1).sum()
num_nio = (all_labels == 2).sum()

total = len(all_labels)

# Compute class weights: inverse frequency
class_weights = [
    total / num_io if num_io > 0 else 0,
    total / num_pseudo if num_pseudo > 0 else 0,
    total / num_nio if num_nio > 0 else 0
]

# --- Train ---
train_model(model, train_loader, val_loader, class_weights, device, epochs=50)

# --- Evaluate ---
evaluate(model, test_loader, device)


