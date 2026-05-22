import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.model import Classifier

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMAGE_SIZE = 128
NUM_CLASSES = 6


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(150),                             # Make sure that the image dimension is 150x150
        transforms.RandomCrop(IMAGE_SIZE),                         # Crops the image to a size 128x128 to be easier to downscale
        transforms.RandomHorizontalFlip(),                  # Random Horiztontal Flips
        transforms.ToTensor(),                              # Min Max normaliztion then rearrange of dimensions(Channels, Height, Width)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],    # Normalization over mean and std
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def get_dataloaders(train_dir, val_dir, batch_size=64, num_workers=2):
    train_transform, val_transform = get_transforms()
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_ds, val_ds, train_dl, val_dl


# Model Training Functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Looping over the dataset batches and performing optimizations
    for images,labels in tqdm(loader,desc='Train',leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()                               # Zeroing the gradient storage
        outputs = model(images)                             # Predicting the labels
        loss = criterion(outputs,labels)                    # Calculating the loss of the batch
        loss.backward()                                     # Back Propagation
        optimizer.step()                                    # Updating the weights

        running_loss += loss.item() * images.size(0)        # Updating the epoch loss with the loss of the current batch (image loss x batch size)
        _,preds = outputs.max(1)                            # Looking for the class index of the highest probability (similar to argmax)
        correct += (preds == labels).sum().item()           # Compares the predictions to the true labels. It counts how many the model got right in this batch and adds it to correct
        total += labels.size(0)                             # Counts how many total images have been processed so far

    epoch_loss = running_loss/total
    epoch_acc = correct/total
    return epoch_loss,epoch_acc

# Model Validation Function
@torch.no_grad()
def validation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images,labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs,labels)

        running_loss += loss.item() * images.size(0)
        _,preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss/total
    epoch_acc = correct/total
    return epoch_loss,epoch_acc


def save_checkpoint(path, model, class_to_idx, best_metric, epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_names': CLASSES,
        'class_to_idx': class_to_idx,
        'image_size': IMAGE_SIZE,
        'architecture_name': 'VGG',
        'best_metric': best_metric,
        'epoch': epoch,
    }
    torch.save(checkpoint, path)


def main(
    train_dir='../data/intel_dataset/seg_train/seg_train',
    val_dir='../data/intel_dataset/seg_test/seg_test',
    epochs=60,
    batch_size=64,
    lr=1e-3,
    weight_decay=1e-3,
    checkpoint_path='../models/best_checkpoint_intel.pth'
):
    device = get_device()

    train_ds, val_ds, train_dl, val_dl = get_dataloaders(
        train_dir,
        val_dir,
        batch_size=batch_size
    )

    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_epoch = -1
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = validation(model, val_dl, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"train_loss {train_loss:.4f}, train_acc {train_acc:.4f} - "
            f"val_loss {val_loss:.4f}, val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_checkpoint(checkpoint_path, model, train_ds.class_to_idx, best_val_acc, best_epoch)

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return history


if __name__ == '__main__':
    main()