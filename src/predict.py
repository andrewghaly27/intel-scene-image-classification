import os
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from src.model import Classifier

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

    images,_ = next(iter(loader))
    images = images.to(device)
    outputs = model(images)
    prob = torch.softmax(outputs, dim=1)
    conf,preds = prob.max(1)

    results = []
    for i in range(len(images)):
        img_cpu = images[i].cpu()
        confidence = conf[i].cpu()
        prediction = preds[i].cpu()
        
        results.append({
            'img': img_cpu,
            'prediction': prediction,
            'confidence': confidence
        })

    return results

# For API Integration
@torch.no_grad()
def predict_pil_image(model, image, device, classes, top_k=3):
    model.eval()
    image = image.convert("RGB")
    transform = get_predict_transform()
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
        results.append({
            "class_name": classes[idx.item()],
            "confidence": prob.item()
        })

    return results


def predict_bytes(model, image_bytes: bytes, device, classes, top_k=3):
    image = Image.open(io.BytesIO(image_bytes))
    return predict_pil_image(model, image, device, classes, top_k=top_k)

# Un-normalize images for visualization
def show(image):
    img = image.numpy()
    img = img.transpose(1,2,0)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img = img*std+mean
    img = np.clip(img,0,1)
    return img


def main(test_dir = '../data/intel_dataset/seg_pred/seg_pred', batch_size=64):
    device = get_device()
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    checkpoint_path = BASE_DIR / "models" / "best_checkpoint_intel.pth"
    
    test_ds = IntelPredictDataset(test_dir,transform=get_predict_transform())
    test_dl = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=2)
    
    model,checkpoint = load_checkpoint(checkpoint_path,device)
    classes = checkpoint.get('class_names',CLASSES)
    
    results = predict(model, test_dl, device)
    fig = plt.figure(figsize=(6,6))
    n=16
    for i in range(n):
        ax = fig.add_subplot(4,4,i+1)
        img = show(results[i]['img'])
        pred_idx = results[i]['prediction'].item()
        conf = results[i]['confidence'].item()
        
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Class: {classes[pred_idx]}\nConfidence: {conf * 100:.2f}%')
    fig.suptitle("Predictions", fontsize = 16)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

