import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models import vgg11, VGG11_Weights
from datasets import FrameVideoDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_frames = 10
n_classes = 10
n_epochs = 10

weights = VGG11_Weights.DEFAULT
ref = weights.transforms()


# --------------------------------------------------
# 1. Early fusion
# --------------------------------------------------

def to_early_fusion(x):
    # [C, T, H, W] -> [T*C, H, W]
    C, Tn, H, W = x.shape
    return x.permute(1, 0, 2, 3).reshape(Tn * C, H, W)


# Applied to individual frames in FrameVideoDataset
frame_tf = T.Compose([
    T.ToTensor()
])

# Applied to the complete stacked video
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.Lambda(to_early_fusion),
    T.Normalize(ref.mean * n_frames, ref.std * n_frames),
])

eval_tf = T.Compose([
    T.Resize((224, 224)),
    T.Lambda(to_early_fusion),
    T.Normalize(ref.mean * n_frames, ref.std * n_frames),
])


# --------------------------------------------------
# 2. Custom collate for video-level transforms
# --------------------------------------------------

class VideoCollate:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        videos, labels = zip(*batch)

        videos = torch.stack([
            self.transform(video) for video in videos
        ])

        labels = torch.tensor(labels, dtype=torch.long)

        return videos, labels


# --------------------------------------------------
# 3. Model
# --------------------------------------------------

def create_model():
    model = vgg11(weights=weights)

    # Save pretrained first convolution
    old_conv = model.features[0]

    # Replace 3 input channels with 30
    new_conv = nn.Conv2d(
        in_channels=3 * n_frames,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    # Initialize using pretrained RGB filters
    with torch.no_grad():
        new_conv.weight.copy_(
            old_conv.weight.repeat(1, n_frames, 1, 1)
            / n_frames
        )

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    # Freeze pretrained convolutional layers
    for p in model.features.parameters():
        p.requires_grad = False

    # Replace first convolution (trainable)
    model.features[0] = new_conv

    # Replace final classification layer
    model.classifier[6] = nn.Linear(4096, n_classes)

    return model.to(device)


# --------------------------------------------------
# 4. Training
# --------------------------------------------------

def train_epoch(model, loader, optimizer, criterion):
    model.train()

    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total, loss_sum / total


# --------------------------------------------------
# 5. Video-level evaluation
# --------------------------------------------------

@torch.no_grad()
def eval_video(model, loader, criterion):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    for x, y in loader:
        # x: [B, 30, 224, 224]
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return correct / total, loss_sum / total


# --------------------------------------------------
# 6. Main
# --------------------------------------------------

if __name__ == "__main__":

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

    testset = FrameVideoDataset(
        split='test',
        transform=frame_tf,
        stack_frames=True
    )

    train_loader = DataLoader(
        trainset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=VideoCollate(train_tf)
    )

    val_loader = DataLoader(
        valset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=VideoCollate(eval_tf)
    )

    test_loader = DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=VideoCollate(eval_tf)
    )

    # Check input shape
    images, labels = next(iter(train_loader))
    print("Input shape:", images.shape)

    model = create_model()

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-4,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    hist = {
        'train_acc': [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }

    best_val = -1.0

    # ----------------------------------------------
    # Training loop
    # ----------------------------------------------

    for epoch in range(1, n_epochs + 1):

        tr_acc, tr_loss = train_epoch(
            model, train_loader, optimizer, criterion
        )

        va_acc, va_loss = eval_video(
            model, val_loader, criterion
        )

        hist['train_acc'].append(tr_acc)
        hist['train_loss'].append(tr_loss)
        hist['val_acc'].append(va_acc)
        hist['val_loss'].append(va_loss)

        print(
            f"epoch {epoch:2d} | "
            f"train loss {tr_loss:.3f} | "
            f"train acc (video) {tr_acc:.3f} | "
            f"val acc (video) {va_acc:.3f}"
        )

        if va_acc > best_val:
            best_val = va_acc
            torch.save(
                model.state_dict(),
                'best_early_fusion.pt'
            )

    # ----------------------------------------------
    # Test best model
    # ----------------------------------------------

    model.load_state_dict(
        torch.load(
            'best_early_fusion.pt',
            map_location=device,
            weights_only=True
        )
    )

    test_acc, test_loss = eval_video(
        model, test_loader, criterion
    )

    print(
        f"best val {best_val:.3f} | "
        f"test acc (video) {test_acc:.3f}"
    )

    # ----------------------------------------------
    # Plot training curves
    # ----------------------------------------------

    epochs = range(1, n_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4)
    )

    ax1.plot(
        epochs, hist['train_acc'],
        'o-', label='train (video-level)'
    )

    ax1.plot(
        epochs, hist['val_acc'],
        'o-', label='val (video-level)'
    )

    ax1.axhline(
        test_acc,
        ls='--',
        c='gray',
        label=f'test (best-val model) = {test_acc:.3f}'
    )

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()
    ax1.grid(alpha=.3)

    ax2.plot(
        epochs, hist['train_loss'],
        'o-', label='train loss'
    )

    ax2.plot(
        epochs, hist['val_loss'],
        'o-', label='val loss'
    )

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend()
    ax2.grid(alpha=.3)

    fig.tight_layout()

    fig.savefig(
        'early_fusion_curves.png',
        dpi=150
    )
