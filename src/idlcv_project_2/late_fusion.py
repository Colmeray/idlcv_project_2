
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
    batch_size=8,
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

        # ResNet18 backbone
        self.cnn = resnet18(weights=None)

        # Remove original classification layer
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        # Classification after feature fusion
        self.classifier = nn.Linear(
            num_features,
            num_classes
        )

    def forward(self, x):

        # Dataset input:
        # (B, C, T, H, W)
        #
        # B = batch size
        # C = RGB channels (3)
        # T = number of video frames
        # H = height
        # W = width

        B, C, T, H, W = x.shape

        if C != 3:
            raise ValueError(
                f"Expected 3 RGB channels, got {C}. "
                f"Input shape: {tuple(x.shape)}"
            )

        # FIX: Change dimension order
        # (B, C, T, H, W)
        # -> (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Combine batch and time dimensions
        # (B, T, C, H, W)
        # -> (B*T, C, H, W)
        x = x.reshape(B * T, C, H, W)

        # Extract features independently
        # from every video frame
        features = self.cnn(x)

        # Output: (B*T, 512)

        # Restore video dimension
        features = features.reshape(B, T, -1)

        # Output: (B, T, 512)

        # Late fusion:
        # Average features across all frames
        features = features.mean(dim=1)

        # Output: (B, 512)

        # Final video classification
        output = self.classifier(features)

        # Output: (B, num_classes)
        return output


# ==========================================
# TRAINING SETUP
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# Check input dimensions
sample_videos, sample_labels = next(iter(train_loader))

print("Input video shape:", sample_videos.shape)
print("Labels shape:", sample_labels.shape)

# Initialize model
model = LateFusionCNN(
    num_classes=10
).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Number of epochs
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

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(videos)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Track loss
        train_loss += (
            loss.item() * labels.size(0)
        )

        # Predictions
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)

    # Training metrics
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

            # Forward pass
            outputs = model(videos)

            # Validation loss
            loss = criterion(outputs, labels)

            val_loss += (
                loss.item() * labels.size(0)
            )

            # Predictions
            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    # Validation metrics
    val_loss /= val_total
    val_acc = val_correct / val_total

    # ======================================
    # PRINT RESULTS
    # ======================================

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
