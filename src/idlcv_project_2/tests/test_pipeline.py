import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import wilson_interval
from models import MODEL_NAMES, make_model
from video_data import UCFVideoDataset, check_split_integrity, get_class_mapping


class PipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for split in ("train", "val", "test"):
            rows = []
            for i, category in enumerate(("PushUps", "PullUps")):
                video = f"v_{split}_{category}_01"
                rows.append({"video_name": video, "label": i, "action": category})
                frame_path = self.root / "frames" / split / category / video
                flow_path = self.root / "flows-png" / split / category / video
                uv_path = self.root / "flows" / split / category / video
                frame_path.mkdir(parents=True)
                flow_path.mkdir(parents=True)
                uv_path.mkdir(parents=True)
                for n in range(1, 11):
                    Image.fromarray(np.full((32, 32, 3), (n * 13) % 255, dtype=np.uint8)).save(frame_path / f"frame_{n}.jpg")
                for n in range(1, 10):
                    Image.fromarray(np.full((32, 32, 3), (n * 15) % 255, dtype=np.uint8)).save(flow_path / f"flow_{n}.png")
                    np.save(uv_path / f"flow_{n}.npy", np.ones((32, 32, 2), dtype="float32"))
            (self.root / "metadata").mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(self.root / "metadata" / f"{split}.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_data_splits_and_frames(self):
        self.assertEqual(check_split_integrity(self.root), {"train": 2, "val": 2, "test": 2})
        mapping, names = get_class_mapping(self.root)
        self.assertEqual(names, ["PushUps", "PullUps"])
        ds = UCFVideoDataset(self.root, "train", mapping, image_size=32)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds[0]["video"].shape, (3, 10, 32, 32))

    def test_optical_flow_formats(self):
        mapping, _ = get_class_mapping(self.root)
        rgb = UCFVideoDataset(self.root, "train", mapping, 32, flow_root=self.root / "flows-png")
        uv = UCFVideoDataset(self.root, "train", mapping, 32, flow_root=self.root / "flows")
        self.assertEqual(rgb.flow_channels, 3)
        self.assertEqual(rgb[0]["flow"].shape, (27, 32, 32))
        self.assertEqual(uv.flow_channels, 2)
        self.assertEqual(uv[0]["flow"].shape, (18, 32, 32))

    def test_models_forward(self):
        frames = torch.randn(1, 3, 10, 32, 32)
        for name in MODEL_NAMES:
            for channels in ((2, 3) if name == "two_stream" else (2,)):
                with self.subTest(name=name, channels=channels):
                    model = make_model(name, num_classes=2, flow_channels=channels).eval()
                    with torch.no_grad():
                        if name == "two_stream":
                            outputs = model(frames, torch.randn(1, 9 * channels, 32, 32))
                        else:
                            outputs = model(frames)
                    self.assertEqual(outputs.shape, (1, 2))

    def test_wilson(self):
        self.assertTrue(np.isnan(wilson_interval(0, 0)[0]))
        self.assertEqual(wilson_interval(0, 4)[0], 0.0)
        self.assertEqual(wilson_interval(4, 4)[1], 1.0)
        self.assertTrue(wilson_interval(3, 4)[0] < 0.75 < wilson_interval(3, 4)[1])


if __name__ == "__main__":
    unittest.main()
