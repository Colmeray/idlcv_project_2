"""Small functions for accuracy confidence intervals and plots."""

import json
import math

import matplotlib
matplotlib.use("Agg")  # Save graphs even when running on a server without a screen.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def wilson_interval(correct, total, confidence=0.95):
    # A 95% Wilson confidence interval for the accuracy.
    # We count VIDEO predictions, not 10 frames separately.
    if total < 0 or correct < 0 or correct > total:
        raise ValueError("correct must be between 0 and total")
    if total == 0:
        return float("nan"), float("nan")
    if confidence != 0.95:
        raise ValueError("This simple version only calculates 95% confidence intervals")

    z = 1.959963984540054  # 95% normal critical value
    accuracy = correct / total
    bottom = 1 + z ** 2 / total
    middle = (accuracy + z ** 2 / (2 * total)) / bottom
    width = z * math.sqrt(accuracy * (1 - accuracy) / total + z ** 2 / (4 * total ** 2)) / bottom
    return max(0, middle - width), min(1, middle + width)


def with_accuracy_ci(correct, total):
    low, high = wilson_interval(correct, total)
    return {"accuracy": correct / total, "correct": correct, "total": total,
            "ci_low": low, "ci_high": high}


def save_history_plot(history, filename, model_name):
    df = pd.DataFrame(history)
    epochs = df["epoch"].to_numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left graph: training and validation loss.
    axs[0].plot(epochs, df["train_loss"], marker="o", label="Train")
    axs[0].plot(epochs, df["val_loss"], marker="o", label="Validation")
    axs[0].set_title("Loss each epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross entropy loss")
    axs[0].legend()

    # Right graph: accuracy, including the validation confidence interval.
    axs[1].plot(epochs, 100 * df["train_accuracy"], marker="o", label="Train")
    axs[1].plot(epochs, 100 * df["val_accuracy"], marker="o", label="Validation")
    axs[1].fill_between(epochs, 100 * df["val_ci_low"].to_numpy(),
                        100 * df["val_ci_high"].to_numpy(), alpha=0.2,
                        label="95% validation CI")
    axs[1].set_title("Accuracy each epoch")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].set_ylim(0, 100)
    axs[1].legend()

    for ax in axs:
        ax.grid(alpha=0.3)
        if len(epochs) <= 20:
            ax.set_xticks(epochs)

    fig.suptitle(model_name)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def save_comparison_plot(results, filename):
    df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
    names = df["model"].tolist()
    accuracies = 100 * df["val_accuracy"].to_numpy()
    low = 100 * df["val_ci_low"].to_numpy()
    high = 100 * df["val_ci_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.8)))
    y = np.arange(len(names))
    ax.barh(y, accuracies)
    ax.errorbar(accuracies, y, xerr=[accuracies - low, high - accuracies],
                fmt="none", ecolor="black", capsize=4)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Best validation accuracy (%)")
    ax.set_title("Model comparison with 95% confidence intervals")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)


def save_confusion_matrix(truth, predicted, class_names, filename):
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    for actual, guess in zip(truth, predicted):
        matrix[actual, guess] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(n), class_names, rotation=40, ha="right")
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test confusion matrix")
    for row in range(n):
        for col in range(n):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > matrix.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    with open(filename.with_suffix(".json"), "w") as file:
        json.dump({"class_names": class_names, "matrix": matrix.tolist()}, file, indent=2)
