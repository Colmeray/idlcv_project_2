"""Train and compare our ResNet18 video classification models.

Example:
python train.py --data-root /path/to/ucf101 --epochs 20 --test-policy none
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import MODEL_NAMES, make_model
from video_data import UCFVideoDataset, check_split_integrity, get_class_mapping
from metrics import with_accuracy_ci, save_history_plot, save_comparison_plot, save_confusion_matrix


def get_args():
    parser = argparse.ArgumentParser(description="UCF101 video classification")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/experiment"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=["frame_avg", "late_fusion", "early_fusion", "r3d_18"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--flow-root", type=Path, default=None)
    parser.add_argument("--flow-format", choices=["auto", "rgb", "uv"], default="auto")
    parser.add_argument("--flow-scale", type=float, default=20.0)
    parser.add_argument("--test-policy", choices=["none", "best", "all"], default="best")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(option):
    if option == "auto":
        if torch.cuda.is_available():
            option = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            option = "mps"
        else:
            option = "cpu"

    if option == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but there is no CUDA GPU")
    if option == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was selected but is not available")
    return torch.device(option)


def get_flow_folder(args):
    if args.flow_root is not None:
        return args.flow_root
    for name in ["flows-png", "flows"]:
        folder = args.data_root / name
        if folder.is_dir():
            return folder
    raise FileNotFoundError("Please set --flow-root for the two_stream model")


def make_dataloader(dataset, args, shuffle, device):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                      generator=generator)


def run_epoch(model, loader, loss_fn, device, optimizer=None, accum_steps=1, use_flow=False):
    # If there is no optimizer we are doing validation/testing, not training.
    training = optimizer is not None
    if training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    actual_labels = []
    predicted_labels = []

    for i, batch in enumerate(loader):
        videos = batch["video"].to(device)
        labels = batch["label"].long().to(device)

        with torch.set_grad_enabled(training):
            if use_flow:
                optical_flow = batch["flow"].to(device)
                scores = model(videos, optical_flow)
            else:
                scores = model(videos)
            loss = loss_fn(scores, labels)

            if training:
                # Small batches can be combined if the GPU has little memory.
                # This also handles an incomplete group of batches at the end.
                group_start = (i // accum_steps) * accum_steps
                group_size = min(accum_steps, len(loader) - group_start)
                (loss / group_size).backward()

                if (i + 1) % accum_steps == 0 or i + 1 == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        guesses = scores.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (guesses == labels).sum().item()
        total += batch_size
        if not training:
            actual_labels.extend(labels.cpu().tolist())
            predicted_labels.extend(guesses.cpu().tolist())

    if total == 0:
        raise ValueError("The dataset contains no examples")

    return {"loss": total_loss / total, "accuracy": correct / total,
            "correct": correct, "total": total,
            "truth": actual_labels, "predicted": predicted_labels}


def make_dataset(args, split, mapping, flow_folder=None):
    return UCFVideoDataset(args.data_root, split, mapping,
                           image_size=args.image_size, frames=args.frames,
                           flow_root=flow_folder, flow_format=args.flow_format,
                           flow_scale=args.flow_scale)


def train_model(args, name, mapping, class_names, device):
    set_seed(args.seed)  # Each model starts with the same random seed.
    folder = args.out_dir / name
    folder.mkdir(parents=True, exist_ok=True)

    use_flow = name == "two_stream"
    flow_folder = get_flow_folder(args) if use_flow else None
    train_data = make_dataset(args, "train", mapping, flow_folder)
    val_data = make_dataset(args, "val", mapping, flow_folder)

    if use_flow and train_data.flow_channels != val_data.flow_channels:
        raise ValueError("Training and validation flow files have different formats")

    train_loader = make_dataloader(train_data, args, True, device)
    val_loader = make_dataloader(val_data, args, False, device)

    model = make_model(name, num_classes=len(mapping), frames=args.frames,
                       flow_channels=train_data.flow_channels or 2,
                       pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_accuracy = -1
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    n_parameters = sum(parameter.numel() for parameter in model.parameters())

    print(f"\nTraining {name} ({n_parameters:,} parameters)", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, loss_fn, device,
                                 optimizer=optimizer, accum_steps=args.accum_steps, use_flow=use_flow)
        val_result = run_epoch(model, val_loader, loss_fn, device, use_flow=use_flow)
        confidence = with_accuracy_ci(val_result["correct"], val_result["total"])

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "val_loss": val_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_accuracy": val_result["accuracy"],
            "val_correct": val_result["correct"],
            "val_total": val_result["total"],
            "val_ci_low": confidence["ci_low"],
            "val_ci_high": confidence["ci_high"]
        }
        history.append(row)

        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        save_history_plot(history, folder / "learning_curves.png", name)

        # Save the model with the best VALIDATION accuracy.
        # Validation loss is used when two epochs have the same accuracy.
        is_best = (row["val_accuracy"] > best_accuracy or
                   (row["val_accuracy"] == best_accuracy and row["val_loss"] < best_loss))
        if is_best:
            best_accuracy = row["val_accuracy"]
            best_loss = row["val_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "state_dict": model.state_dict(),
                "model": name,
                "epoch": epoch,
                "num_classes": len(mapping),
                "frames": args.frames,
                "flow_channels": train_data.flow_channels,
                "label_mapping": mapping,
                "class_names": class_names
            }, folder / "best.pt")
        else:
            epochs_without_improvement += 1

        print(f"{name} | epoch {epoch} | "
              f"train loss {row['train_loss']:.3f}, accuracy {row['train_accuracy']:.1%} | "
              f"val loss {row['val_loss']:.3f}, accuracy {row['val_accuracy']:.1%} | "
              f"95% CI [{row['val_ci_low']:.1%}, {row['val_ci_high']:.1%}]", flush=True)

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping. Best epoch was {best_epoch}.")
            break

    # Look up the best epoch for the comparison table.
    best_row = history[best_epoch - 1]
    return {
        "model": name,
        "best_epoch": best_epoch,
        "val_accuracy": best_row["val_accuracy"],
        "val_correct": best_row["val_correct"],
        "val_total": best_row["val_total"],
        "val_ci_low": best_row["val_ci_low"],
        "val_ci_high": best_row["val_ci_high"],
        "val_loss": best_row["val_loss"],
        "params": n_parameters,
        "test_accuracy": float("nan"),
        "test_ci_low": float("nan"),
        "test_ci_high": float("nan")
    }


def test_model(args, result_row, mapping, class_names, device):
    name = result_row["model"]
    folder = args.out_dir / name
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)

    flow_folder = get_flow_folder(args) if name == "two_stream" else None
    test_data = make_dataset(args, "test", mapping, flow_folder)
    test_loader = make_dataloader(test_data, args, False, device)

    if name == "two_stream" and test_data.flow_channels != checkpoint["flow_channels"]:
        raise ValueError("Test flow format does not match the trained model")

    # We don't download weights here; we load our saved checkpoint instead.
    model = make_model(name, len(mapping), args.frames,
                       checkpoint["flow_channels"] or 2, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loss_fn = nn.CrossEntropyLoss()
    test_result = run_epoch(model, test_loader, loss_fn, device, use_flow=(name == "two_stream"))
    confidence = with_accuracy_ci(test_result["correct"], test_result["total"])

    result_row["test_accuracy"] = confidence["accuracy"]
    result_row["test_ci_low"] = confidence["ci_low"]
    result_row["test_ci_high"] = confidence["ci_high"]
    result_row["test_loss"] = test_result["loss"]
    result_row["test_correct"] = confidence["correct"]
    result_row["test_total"] = confidence["total"]

    save_confusion_matrix(test_result["truth"], test_result["predicted"],
                          class_names, folder / "confusion_matrix_test.png")
    with open(folder / "test_metrics.json", "w") as file:
        json.dump({"model": name, "best_validation_epoch": checkpoint["epoch"],
                   "test": {"loss": test_result["loss"], **confidence}}, file, indent=2)

    print(f"Final test {name}: {confidence['accuracy']:.1%} "
          f"(95% CI [{confidence['ci_low']:.1%}, {confidence['ci_high']:.1%}])")


def main():
    args = get_args()
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.frames < 2:
        raise ValueError("Use at least 1 epoch/batch/accum-step and at least 2 frames")
    if args.image_size < 32 or args.lr <= 0 or args.flow_scale <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid image size, learning rate, flow scale or weight decay")
    if len(args.models) != len(set(args.models)):
        raise ValueError("Do not put the same model name twice")

    set_seed(args.seed)
    device = get_device(args.device)
    counts = check_split_integrity(args.data_root)
    mapping, class_names = get_class_mapping(args.data_root)
    if len(mapping) != 10:
        print(f"Note: found {len(mapping)} classes instead of the 10 from the assignment.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Save the settings so we can remember how the experiment was run.
    settings = vars(args).copy()
    settings["data_root"] = str(args.data_root)
    settings["out_dir"] = str(args.out_dir)
    settings["flow_root"] = str(args.flow_root) if args.flow_root else None
    with open(args.out_dir / "config.json", "w") as file:
        json.dump({"settings": settings, "device": str(device),
                   "split_sizes": counts, "label_mapping": mapping,
                   "class_names": class_names}, file, indent=2)

    print("Using device:", device)
    print("Videos in each split:", counts)
    if args.pretrained:
        print("Note: 2D weights come from ImageNet and 3D weights from Kinetics, so pretraining differs.")

    results = []
    for name in args.models:
        results.append(train_model(args, name, mapping, class_names, device))

    # Decide which model wins using validation only, never test accuracy.
    best_result = sorted(results, key=lambda r: (-r["val_accuracy"], r["val_loss"]))[0]
    winner = best_result["model"]
    save_comparison_plot(results, args.out_dir / "validation_comparison.png")
    print("Best validation model:", winner)

    if args.test_policy == "best":
        test_model(args, best_result, mapping, class_names, device)
    elif args.test_policy == "all":
        # Only use this for the final report, after choosing all hyperparameters.
        for result in results:
            test_model(args, result, mapping, class_names, device)

    pd.DataFrame(results).to_csv(args.out_dir / "comparison.csv", index=False)
    print("Results saved in:", args.out_dir)


if __name__ == "__main__":
    main()
