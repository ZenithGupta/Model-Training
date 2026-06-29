"""
Plot Training Curves — Qwen3-14B Trigonometry Fine-Tuning
=========================================================

Generates a professional training loss visualization.

Usage:
    python plot_training.py
    python plot_training.py --output ../assets/training_curves.png
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]


def plot_training_curves(output_path: str = "../assets/training_curves.png"):
    """Generate professional training curves visualization."""
    
    # Training data
    steps = [75, 150, 225, 300, 375, 450, 525, 600]
    train_loss = [0.497, 0.472, 0.455, 0.435, 0.474, 0.469, 0.418, 0.411]
    val_loss = [0.487, 0.476, 0.472, 0.469, 0.467, 0.466, 0.465, 0.464]
    
    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    
    # Plot lines
    ax.plot(
        steps, train_loss,
        color="#58a6ff", linewidth=2.5, marker="o", markersize=8,
        label="Training Loss", zorder=5,
        markerfacecolor="#58a6ff", markeredgecolor="white", markeredgewidth=1.5,
    )
    ax.plot(
        steps, val_loss,
        color="#f78166", linewidth=2.5, marker="s", markersize=8,
        label="Validation Loss", zorder=5,
        markerfacecolor="#f78166", markeredgecolor="white", markeredgewidth=1.5,
    )
    
    # Fill between
    ax.fill_between(steps, train_loss, val_loss, alpha=0.1, color="#58a6ff")
    
    # Annotations
    ax.annotate(
        f"Final: {val_loss[-1]:.3f}",
        xy=(steps[-1], val_loss[-1]),
        xytext=(steps[-1] - 80, val_loss[-1] + 0.012),
        fontsize=11, color="#f78166", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#f78166", lw=1.5),
    )
    ax.annotate(
        f"Final: {train_loss[-1]:.3f}",
        xy=(steps[-1], train_loss[-1]),
        xytext=(steps[-1] - 80, train_loss[-1] - 0.015),
        fontsize=11, color="#58a6ff", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.5),
    )
    
    # Styling
    ax.set_xlabel("Training Steps", fontsize=13, color="white", labelpad=10)
    ax.set_ylabel("Loss", fontsize=13, color="white", labelpad=10)
    ax.set_title(
        "Qwen3-14B Trigonometry LoRA — Training Curves",
        fontsize=16, color="white", fontweight="bold", pad=20,
    )
    
    ax.tick_params(colors="white", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.grid(True, alpha=0.15, color="white", linestyle="--")
    
    legend = ax.legend(
        fontsize=12, loc="upper right",
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white",
    )
    
    ax.set_xlim(50, 625)
    ax.set_ylim(0.39, 0.51)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    print(f"✅ Training curves saved to: {output_path}")
    plt.close()


def plot_comparison_bar(output_path: str = "../assets/comparison_bar.png"):
    """Generate comparison bar chart."""
    
    categories = [
        "Problems\nAnswered",
        "Correct\nAnswers",
        "Structured\nOutput",
        "Token\nEfficient",
        "Production\nReady",
    ]
    fine_tuned = [5, 5, 5, 5, 5]
    base = [0, 0, 0, 0, 0]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(
        x - width / 2, fine_tuned, width,
        label="Fine-Tuned", color="#3fb950", edgecolor="white", linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2, base, width,
        label="Base Model", color="#f85149", edgecolor="white", linewidth=0.5,
    )
    
    # Add value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            "5/5", ha="center", va="bottom", color="#3fb950",
            fontweight="bold", fontsize=12,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2, 0.15,
            "0/5", ha="center", va="bottom", color="#f85149",
            fontweight="bold", fontsize=12,
        )
    
    ax.set_ylabel("Score (out of 5)", fontsize=13, color="white", labelpad=10)
    ax.set_title(
        "Fine-Tuned vs Base Model — Head-to-Head Comparison",
        fontsize=16, color="white", fontweight="bold", pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, color="white")
    ax.tick_params(colors="white", labelsize=11)
    ax.set_ylim(0, 6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.grid(True, axis="y", alpha=0.15, color="white", linestyle="--")
    
    legend = ax.legend(
        fontsize=12, loc="upper right",
        facecolor="#161b22", edgecolor="#30363d",
        labelcolor="white",
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="#0d1117")
    print(f"✅ Comparison chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training visualizations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../assets",
        help="Output directory for plots",
    )
    args = parser.parse_args()
    
    plot_training_curves(f"{args.output_dir}/training_curves.png")
    plot_comparison_bar(f"{args.output_dir}/comparison_bar.png")
