from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_metadata(root, split):
    filename = Path(root) / "metadata" / f"{split}.csv"
    if not filename.exists():
        raise FileNotFoundError(f"Cannot find {filename}")

    data = pd.read_csv(filename)
    if "video_name" not in data or "label" not in data:
        raise ValueError(f"Missing video_name or label column in {filename}")
    if data[["video_name", "label"]].isna().any().any():
        raise ValueError(f"Missing video names or labels in {filename}")

    data = data.copy()
    data["video_name"] = data["video_name"].apply(lambda name: Path(str(name)).stem)
    data["label_text"] = data["label"].apply(lambda label: str(label).strip())
    if data["video_name"].duplicated().any():
        raise ValueError(f"Duplicate video names in {filename}")
    return data


def get_class_mapping(root):
    data = read_metadata(root, "train")
    labels = list(data["label_text"].unique())

    # Numeric labels are normally 0,1,...,9.
    if all(label.lstrip("-").isdigit() for label in labels):
        labels = sorted(labels, key=int)
    else:
        labels = sorted(labels)

    mapping = {}
    class_names = []
    for number, label in enumerate(labels):
        mapping[label] = number
        class_names.append("Class " + label)

    # Use the activity names if the CSV contains them.
    for column in ["action", "actions", "class_name", "class", "category"]:
        if column in data:
            if all(data[data["label_text"] == label][column].nunique() == 1 for label in labels):
                class_names = [str(data[data["label_text"] == label][column].iloc[0]) for label in labels]
                break
    return mapping, class_names


def check_split_integrity(root):
    splits = {}
    for split in ["train", "val", "test"]:
        splits[split] = read_metadata(root, split)

    # We should not have the same video in two groups.
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        duplicates = set(splits[first]["video_name"]) & set(splits[second]["video_name"])
        if duplicates:
            raise ValueError(f"Some videos occur in both {first} and {second}: {sorted(duplicates)[:5]}")

    # This only detects the same people if the CSV actually includes their IDs.
    for column in ["subject_id", "person_id", "participant_id", "identity_id"]:
        if all(column in splits[split] for split in splits):
            for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
                first_ids = set(splits[first][column].dropna())
                second_ids = set(splits[second][column].dropna())
                if first_ids & second_ids:
                    raise ValueError(f"Same person appears in {first} and {second}")
            break

    return {split: len(splits[split]) for split in splits}


def sorted_files(folder, allowed_extensions):
    files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in allowed_extensions:
            files.append(file)

    # Make sure frame_2 comes before frame_10.
    def file_number(file):
        numbers = re.findall(r"\d+", file.stem)
        return int(numbers[-1]) if numbers else 0

    return sorted(files, key=lambda file: (file_number(file), file.name))


def find_video_folders(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Cannot find folder: {folder}")

    folders = {}
    # Expected structure: split / activity / video_name / files
    for activity_folder in folder.iterdir():
        if not activity_folder.is_dir():
            continue
        for video_folder in activity_folder.iterdir():
            if video_folder.is_dir():
                if video_folder.name in folders:
                    raise ValueError(f"Repeated video folder name: {video_folder.name}")
                folders[video_folder.name] = video_folder
    return folders


class UCFVideoDataset(Dataset):
    def __init__(self, root, split, mapping, image_size=112, frames=10,
                 flow_root=None, flow_format="auto", flow_scale=20.0):
        self.root = Path(root)
        self.split = split
        self.frames = frames
        self.image_size = image_size
        self.flow_scale = flow_scale
        self.flow_channels = 0

        if flow_format not in ["auto", "uv", "rgb"]:
            raise ValueError("flow_format should be auto, uv or rgb")
        if flow_scale <= 0:
            raise ValueError("flow_scale must be positive")

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

        data = read_metadata(self.root, split)
        rgb_folders = find_video_folders(self.root / "frames" / split)
        flow_folders = None
        if flow_root is not None:
            flow_folders = find_video_folders(Path(flow_root) / split)

        self.samples = []
        for _, row in data.iterrows():
            name = row["video_name"]
            if row["label_text"] not in mapping:
                raise ValueError(f"Unknown class in {split}: {row['label_text']}")
            label = mapping[row["label_text"]]

            if name not in rgb_folders:
                raise FileNotFoundError(f"Could not find RGB frames for {name}")

            frame_paths = sorted_files(rgb_folders[name], {".jpg", ".jpeg", ".png"})
            frame_paths = [p for p in frame_paths if re.fullmatch(r"frame_\d+", p.stem)]
            if len(frame_paths) != frames:
                raise ValueError(f"{name} has {len(frame_paths)} frames, expected {frames}")

            flow_paths = None
            if flow_folders is not None:
                if name not in flow_folders:
                    raise FileNotFoundError(f"Could not find optical flow for {name}")
                flow_paths = sorted_files(flow_folders[name], {".png", ".jpg", ".jpeg", ".npy", ".npz", ".pt"})
                if len(flow_paths) != frames - 1:
                    raise ValueError(f"{name} should have {frames - 1} flow files")

                formats = set()
                for path in flow_paths:
                    formats.add("uv" if path.suffix.lower() in [".npy", ".npz", ".pt"] else "rgb")
                if len(formats) != 1:
                    raise ValueError(f"Different flow formats were mixed for {name}")
                current_format = formats.pop()
                if flow_format != "auto" and flow_format != current_format:
                    raise ValueError(f"Expected {flow_format} flow but found {current_format}")

                channels = 2 if current_format == "uv" else 3
                if self.flow_channels not in [0, channels]:
                    raise ValueError("Flow formats must match for all videos")
                self.flow_channels = channels

            self.samples.append((name, label, frame_paths, flow_paths))

        if len(self.samples) == 0:
            raise ValueError(f"No videos found in {split}")

    def __len__(self):
        return len(self.samples)

    def load_rgb(self, filename):
        with Image.open(filename) as image:
            return self.image_transform(image.convert("RGB"))

    def load_uv(self, filename):
        # Numerical optical flow consists of horizontal and vertical motion.
        if filename.suffix.lower() == ".npy":
            array = np.load(filename, allow_pickle=False)
        elif filename.suffix.lower() == ".npz":
            with np.load(filename, allow_pickle=False) as saved:
                key = "flow" if "flow" in saved else saved.files[0]
                array = saved[key]
        else:
            array = torch.load(filename, map_location="cpu", weights_only=True)
            if isinstance(array, dict):
                array = array["flow"] if "flow" in array else next(iter(array.values()))

        flow = torch.as_tensor(array, dtype=torch.float32)
        if flow.ndim != 3:
            raise ValueError(f"Flow should be [2,H,W] or [H,W,2]: {filename}")
        if flow.shape[0] == 2:
            pass
        elif flow.shape[-1] == 2:
            flow = flow.permute(2, 0, 1)
        else:
            raise ValueError(f"Flow must have two components: {filename}")
        if not torch.isfinite(flow).all():
            raise ValueError(f"Flow has invalid values: {filename}")

        old_height, old_width = flow.shape[1], flow.shape[2]
        flow = F.interpolate(flow.unsqueeze(0), size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)[0]
        # Flow represents pixel movement, so resize the movement too.
        flow[0] *= self.image_size / old_width
        flow[1] *= self.image_size / old_height
        return torch.clamp(flow / self.flow_scale, -1, 1)

    def __getitem__(self, index):
        name, label, frame_paths, flow_paths = self.samples[index]

        images = []
        for path in frame_paths:
            images.append(self.load_rgb(path))
        video = torch.stack(images, dim=1)  # [3,10,H,W]

        result = {"video": video, "label": label, "video_name": name}

        if flow_paths is not None:
            flows = []
            for path in flow_paths:
                if self.flow_channels == 2:
                    flows.append(self.load_uv(path))
                else:
                    # PNG optical flow may be a color visualization, not raw u/v!
                    flows.append(self.load_rgb(path))
            result["flow"] = torch.cat(flows, dim=0)
        return result
