import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.efficientnet import MBConv, MBConvConfig
from datasets import FrameImageDataset, FrameVideoDataset
from torchvision.io import decode_image
from torchvision.models import vgg11, VGG11_Weights


device = torch.device('cuda')


weights = VGG11_Weights.DEFAULT
ref = weights.transforms()          # indeholder ImageNet mean/std

# Samme normalisering som VGG blev trænet med — vigtigt!
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(ref.mean, ref.std),
])
eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(ref.mean, ref.std),
])

# batch_size = 1
# trainset = FrameImageDataset(split="train")
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testset = FrameVideoDataset(split="test")
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

trainset = FrameImageDataset(split='train', transform=train_tf)
valset   = FrameVideoDataset(split='val',   transform=eval_tf, stack_frames=True)
train_loader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(valset,   batch_size=4,  shuffle=False, num_workers=4)


model = vgg11(weights=weights)
for p in model.features.parameters():
    p.requires_grad = False                 # frys conv-delen
model.classifier[6] = nn.Linear(4096, 10)   # 1000 ImageNet-klasser -> 10
model = model.to(device)

optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader):
    model.train()                       # slår dropout til i classifier
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        i += 1
    return correct / total              # frame-accuracy

@torch.no_grad()
def eval_video(model, loader):
    model.eval()
    correct = 0
    for frames, y in loader:                     # frames: [B, C, T, H, W]
        B, C, Tn, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * Tn, C, H, W).to(device)
        probs = model(x).softmax(1).view(B, Tn, -1).mean(1)   # gennemsnit over T frames
        correct += (probs.argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)

print(train_epoch(model, train_loader))
print(eval_video(model, val_loader))

