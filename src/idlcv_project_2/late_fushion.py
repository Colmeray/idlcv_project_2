
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import resnet18

from datasets import FrameVideoDataset


# ==========================================
# TRANSFORMS
# ==========================================

# Applied to each individual frame
frame_tf = T.Compose([
    T.ToTensor(),
])


# ==========================================
# COLLATE FUNCTIONS
# ==========================================

def train_collate(batch):
    videos, labels = zip(*batch)
    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels


def eval_collate(batch):
    videos, labels = zip(*batch)
    videos = torch.stack(videos)
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels


# ==========================================
# DATASETS
# ==========================================

trainset = FrameVideoDataset(
    split="train",
    transform=frame_tf,
    stack_frames=True
)

valset = FrameVideoDataset(
    split="val",
    transform=frame_tf,
    stack_frames=True
)


# ==========================================
# DATALOADERS
# ==========================================

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


# ==========================================
# LATE FUSION CNN
# ==========================================

class LateFusionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.cnn = resnet18(weights=None)

        # Remove the original classification layer
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # Classification after fusion
        self.classifier = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Process each frame independently
        x = x.reshape(B * T, C, H, W)

        # Extract features
        features = self.cnn(x)
        # (B*T, 512)

        # Restore video dimension
        features = features.reshape(B, T, -1)
        # (B, T, 512)

        # Late fusion: average over frames
        features = features.mean(dim=1)
        # (B, 512)

        # Final classification
        output = self.classifier(features)
        # (B, num_classes)

        return output


# ==========================================
# TRAINING SETUP
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = LateFusionCNN(num_classes=10).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10


# ==========================================
# TRAINING LOOP
# ==========================================

for epoch in range(epochs):

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for videos, labels in train_loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for videos, labels in val_loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(
    model.state_dict(),
    "late_fusion_cnn.pth"
)

print("Model saved!")
