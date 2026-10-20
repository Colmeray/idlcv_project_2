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
ref = weights.transforms()

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

testset     = FrameVideoDataset(split='test', transform=eval_tf, stack_frames=True)
test_loader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

def train_epoch(model, loader):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total      # frame-accuracy, frame-loss

@torch.no_grad()
def eval_video(model, loader):
    model.eval()
    correct = 0
    for frames, y in loader:                     # [B, C, T, H, W]
        B, C, Tn, H, W = frames.shape
        x = frames.permute(0, 2, 1, 3, 4).reshape(B * Tn, C, H, W).to(device)
        probs = model(x).softmax(1).view(B, Tn, -1).mean(1)
        correct += (probs.argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)

n_epochs = 10
hist = {'train_acc': [], 'train_loss': [], 'val_acc': []}
best_val = 0.0

for epoch in range(1, n_epochs + 1):
    tr_acc, tr_loss = train_epoch(model, train_loader)
    va_acc = eval_video(model, val_loader)
    hist['train_acc'].append(tr_acc)
    hist['train_loss'].append(tr_loss)
    hist['val_acc'].append(va_acc)
    print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} | "
          f"train acc (frame) {tr_acc:.3f} | val acc (video) {va_acc:.3f}")
    if va_acc > best_val:                        # gem bedste model efter val
        best_val = va_acc
        torch.save(model.state_dict(), 'best_per_frame.pt')

# Test KUN én gang, på den model val valgte
model.load_state_dict(torch.load('best_per_frame.pt'))
test_acc = eval_video(model, test_loader)
print(f"best val {best_val:.3f} | test acc (video) {test_acc:.3f}")

epochs = range(1, n_epochs + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(epochs, hist['train_acc'], 'o-', label='train (frame-level)')
ax1.plot(epochs, hist['val_acc'],   'o-', label='val (video-level, mean softmax)')
ax1.axhline(test_acc, ls='--', c='gray', label=f'test (best-val model) = {test_acc:.3f}')
ax1.set_xlabel('epoch'); ax1.set_ylabel('accuracy'); ax1.legend(); ax1.grid(alpha=.3)
ax2.plot(epochs, hist['train_loss'], 'o-')
ax2.set_xlabel('epoch'); ax2.set_ylabel('train loss'); ax2.grid(alpha=.3)
fig.tight_layout()
fig.savefig('per_frame_curves.png', dpi=150)

