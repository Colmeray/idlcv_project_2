"""The different ResNet18 models used for our video project.

A video has the shape [batch, 3, 10, height, width].
The optical flow is only needed for the two_stream model.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights


MODEL_NAMES = ["frame_avg", "late_fusion", "early_fusion", "r3d_18", "two_stream"]


def get_resnet(pretrained=False):
    # We remove the final classification layer so we get 512 features.
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    net = resnet18(weights=weights)
    net.fc = nn.Identity()
    return net


def change_input_channels(net, channels):
    # ResNet18 normally expects 3 RGB channels. A stack of images has more.
    old_conv = net.conv1
    new_conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Reuse the original weights, including when using pretrained ResNet18.
    with torch.no_grad():
        if channels % 3 == 0:
            new_weights = old_conv.weight.repeat(1, channels // 3, 1, 1)
            new_weights = new_weights / (channels // 3)
        else:
            new_weights = old_conv.weight.mean(dim=1, keepdim=True)
            new_weights = new_weights.repeat(1, channels, 1, 1) * 3 / channels
        new_conv.weight.copy_(new_weights)

    net.conv1 = new_conv


class FrameAverageResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.resnet = get_resnet(pretrained)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, video):
        batch, channels, frames, height, width = video.shape

        # Put all the frames in one batch, so ResNet can process them.
        images = video.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch * frames, channels, height, width)
        predictions = self.fc(self.resnet(images))

        # Average the predictions from the 10 frames.
        predictions = predictions.reshape(batch, frames, -1)
        return predictions.mean(dim=1)


class LateFusionResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        self.resnet = get_resnet(pretrained)
        self.attention = nn.Linear(512, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, video):
        batch, channels, frames, height, width = video.shape
        images = video.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch * frames, channels, height, width)

        # Here we join the frame features at the end of the network.
        features = self.resnet(images).reshape(batch, frames, 512)
        weights = torch.softmax(self.attention(features), dim=1)
        combined_features = (features * weights).sum(dim=1)
        return self.fc(combined_features)


class EarlyFusionResNet18(nn.Module):
    def __init__(self, num_classes=10, frames=10, pretrained=False):
        super().__init__()
        self.frames = frames
        self.resnet = get_resnet(pretrained)
        change_input_channels(self.resnet, frames * 3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, video):
        batch, channels, frames, height, width = video.shape
        if channels != 3 or frames != self.frames:
            raise ValueError("Early fusion expects 10 RGB frames (or the --frames setting).")

        # Stack the frames as channels before sending them into ResNet.
        images = video.permute(0, 2, 1, 3, 4)
        images = images.reshape(batch, frames * 3, height, width)
        return self.fc(self.resnet(images))


class VideoResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        self.resnet = r3d_18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, video):
        return self.resnet(video)


class TwoStreamResNet18(nn.Module):
    def __init__(self, num_classes=10, frames=10, flow_channels=2, pretrained=False):
        super().__init__()
        if flow_channels not in (2, 3):
            raise ValueError("Flow must have 2 channels (u,v) or 3 (RGB flow images).")

        self.frames = frames
        self.flow_channels = flow_channels
        self.rgb_resnet = get_resnet(pretrained)
        self.flow_resnet = get_resnet(pretrained)
        change_input_channels(self.flow_resnet, (frames - 1) * flow_channels)
        self.rgb_fc = nn.Linear(512, num_classes)
        self.flow_fc = nn.Linear(512, num_classes)

    def forward(self, video, flow):
        if video.shape[2] != self.frames:
            raise ValueError("Incorrect number of RGB frames.")
        expected = (self.frames - 1) * self.flow_channels
        if flow.ndim != 4 or flow.shape[1] != expected:
            raise ValueError(f"Expected {expected} channels of optical flow.")

        # One RGB frame goes into one ResNet.
        middle_frame = video[:, :, self.frames // 2]
        rgb_scores = self.rgb_fc(self.rgb_resnet(middle_frame))

        # All 9 optical-flow images go into the other ResNet.
        flow_scores = self.flow_fc(self.flow_resnet(flow))

        # Combine the scores from both streams.
        return (rgb_scores + flow_scores) / 2


def make_model(name, num_classes=10, frames=10, flow_channels=2, pretrained=False):
    if name == "frame_avg":
        return FrameAverageResNet18(num_classes, pretrained)
    elif name == "late_fusion":
        return LateFusionResNet18(num_classes, pretrained)
    elif name == "early_fusion":
        return EarlyFusionResNet18(num_classes, frames, pretrained)
    elif name == "r3d_18":
        return VideoResNet18(num_classes, pretrained)
    elif name == "two_stream":
        return TwoStreamResNet18(num_classes, frames, flow_channels, pretrained)
    else:
        raise ValueError(f"Unknown model: {name}")
