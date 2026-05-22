import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.model import Classifier

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMAGE_SIZE = 128
NUM_CLASSES = 6

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = Classifier().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


@torch.no_grad()
def evaluate_model(model, loader, classes, device):
    model.eval()
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, preds = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = (all_labels == all_preds).mean()
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report)

    # Display Confusion Matrix
    plt.figure(figsize=(8,8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy, report, cm


def main(val_dir='../data/intel_dataset/seg_test/seg_test', batch_size=64):
    device = get_device()

    BASE_DIR = Path(__file__).resolve().parent.parent
    checkpoint_path = BASE_DIR / "models" / "best_checkpoint_intel.pth"
    
    val_ds = datasets.ImageFolder(val_dir, transform=get_eval_transform())
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model, checkpoint = load_checkpoint(checkpoint_path, device)
    classes = checkpoint.get('class_names', CLASSES)

    return evaluate_model(model, val_dl, classes, device)


if __name__ == '__main__':
    main()
