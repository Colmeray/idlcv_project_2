import os
import numpy as np

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

def to_early_fusion(x):
    C, Tn, H, W = x.shape
    return x.permute(1, 0, 2, 3).reshape(Tn * C, H, W)


# Applied to individual frames by FrameVideoDataset
frame_tf = T.Compose([
    T.ToTensor(),
])

# Applied to complete videos
train_tf = T.Compose([
    T.Lambda(to_early_fusion)
])

eval_tf = T.Compose([
    T.Lambda(to_early_fusion)
])


def train_collate(batch):
    videos, labels = zip(*batch)
    videos = torch.stack([train_tf(video) for video in videos])
    return videos, torch.tensor(labels)


def eval_collate(batch):
    videos, labels = zip(*batch)
    videos = torch.stack([eval_tf(video) for video in videos])
    return videos, torch.tensor(labels)


trainset = FrameVideoDataset(
    split='train',
    transform=frame_tf,
    stack_frames=True
)

valset = FrameVideoDataset(
    split='val',
    transform=frame_tf,
    stack_frames=True
)

train_loader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    collate_fn=train_collate
)

val_loader = DataLoader(
    valset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=eval_collate
)

images, labels = next(iter(train_loader))
print(images.shape)





