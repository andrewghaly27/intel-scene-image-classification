import torch
import torch.nn as nn

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