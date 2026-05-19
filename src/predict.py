import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMAGE_SIZE = 128
NUM_CLASSES = 6


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntelPredictDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # List all image files in the directory
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image and ensure it's RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Returning img_name helps identify which file matches which prediction
        return image, img_name
    

# Building Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        # 128 x 128 -> 64 x 64
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # 64 x 64 -> 32 x 32
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # 32 x 32 -> 16 x 16
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        # 16 x 16 -> 8 x 8
        self.block4 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,padding='same',bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512,1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(1000,1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(1000,6)
        )

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.gap(x)
        x=self.flatten(x)
        x=self.classifier(x)
        return x


def get_predict_transform():
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


# Model Prediction Function
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    img = []

    images,_ = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    conf,preds = outputs.max(1)

    for i in range(len(images)):
        img.append(images[i].cpu())

    return img, preds, conf


def show(image):
    img = image.numpy()
    img = img.transpose(1,2,0)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img*std+mean
    img = np.clip(img,0,1)
    return img


def main(test_dir = '../data/intel_dataset/seg_pred/seg_pred', checkpoint_path='../models/best_checkpoint_intel.pth',batch_size=64):
    device = get_device()
    
    test_ds = IntelPredictDataset(test_dir,transform=get_predict_transform())
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=2)
    
    model,checkpoint = load_checkpoint(checkpoint_path,device)
    classes = checkpoint.get('class_names',CLASSES)
    
    pred_images, pred_labels, confidence = predict(model, test_dl, device)
    fig = plt.figure(figsize=(6,6))
    n=16
    for i in range(n):
        ax = fig.add_subplot(4,4,i+1)
        img = show(pred_images[i])
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Class: {classes[pred_labels[i]]}\nConfidence: {confidence[i]}')
    fig.suptitle("Predictions", fontsize = 16)
    fig.tight_layout()


if __name__ == '__main__':
    main()