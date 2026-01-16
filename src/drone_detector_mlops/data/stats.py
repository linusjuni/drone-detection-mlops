"""Dataset statistics and visualization module."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from drone_detector_mlops.data.data import DroneVsBirdDataset

app = typer.Typer()

LABEL_NAMES = {0: "drone", 1: "bird"}


def show_sample_images(dataset: DroneVsBirdDataset, n_samples: int = 16) -> plt.Figure:
    """Create a grid of sample images from the dataset."""
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    indices = torch.randperm(len(dataset))[:n_samples]

    for ax_idx, data_idx in enumerate(indices):
        # Get raw image without transforms
        img_path = dataset._build_image_path(data_idx.item())
        image = dataset._open_image(img_path)
        label = dataset.labels[data_idx.item()]

        axes[ax_idx].imshow(image)
        axes[ax_idx].set_title(f"{LABEL_NAMES[label]}", fontsize=10)
        axes[ax_idx].axis("off")

    # Hide empty subplots
    for ax_idx in range(n_samples, len(axes)):
        axes[ax_idx].axis("off")

    fig.tight_layout()
    return fig


def plot_label_distribution(labels: list[int], title: str) -> plt.Figure:
    """Plot label distribution as a bar chart."""
    label_counts = torch.bincount(torch.tensor(labels), minlength=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2ecc71", "#3498db"]  # green for drone, blue for bird
    bars = ax.bar(
        [LABEL_NAMES[0], LABEL_NAMES[1]], label_counts.numpy(), color=colors, edgecolor="black", linewidth=1.2
    )

    # Add count labels on bars
    for bar, count in zip(bars, label_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(count.item()), ha="center", fontsize=12)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


def get_image_stats(dataset: DroneVsBirdDataset, n_samples: int = 50) -> dict:
    """Compute image statistics from a sample of images."""
    widths, heights = [], []
    indices = torch.randperm(len(dataset))[:n_samples]

    for idx in indices:
        img_path = dataset._build_image_path(idx.item())
        image = dataset._open_image(img_path)
        widths.append(image.size[0])
        heights.append(image.size[1])

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": sum(widths) / len(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": sum(heights) / len(heights),
    }


def print_markdown_report(
    train_dataset: DroneVsBirdDataset,
    val_dataset: DroneVsBirdDataset,
    test_dataset: DroneVsBirdDataset,
    img_stats: dict,
) -> None:
    """Print dataset statistics in markdown format."""
    print("# 📊 Drone vs Bird Dataset Statistics\n")

    # Split sizes table
    print("## Split Sizes\n")
    print("| Split | Images |")
    print("|-------|--------|")
    print(f"| Train | {len(train_dataset)} |")
    print(f"| Validation | {len(val_dataset)} |")
    print(f"| Test | {len(test_dataset)} |")
    print(f"| **Total** | **{len(train_dataset) + len(val_dataset) + len(test_dataset)}** |")
    print()

    # Class distribution table
    print("## Class Distribution\n")
    print("| Split | Drone | Bird | Drone % | Bird % |")
    print("|-------|-------|------|---------|--------|")

    for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        label_counts = torch.bincount(torch.tensor(dataset.labels), minlength=2)
        drone_pct = 100 * label_counts[0].item() / len(dataset)
        bird_pct = 100 * label_counts[1].item() / len(dataset)
        print(f"| {name} | {label_counts[0].item()} | {label_counts[1].item()} | {drone_pct:.1f}% | {bird_pct:.1f}% |")
    print()

    # Image statistics
    print("## Image Statistics (sampled)\n")
    print("| Dimension | Min | Max | Avg |")
    print("|-----------|-----|-----|-----|")
    print(f"| Width | {img_stats['min_width']:.0f} | {img_stats['max_width']:.0f} | {img_stats['avg_width']:.1f} |")
    print(f"| Height | {img_stats['min_height']:.0f} | {img_stats['max_height']:.0f} | {img_stats['avg_height']:.1f} |")
    print()


def print_console_report(
    train_dataset: DroneVsBirdDataset,
    val_dataset: DroneVsBirdDataset,
    test_dataset: DroneVsBirdDataset,
    img_stats: dict,
    data_dir: Path,
    splits_dir: Path,
) -> None:
    """Print dataset statistics in console format."""
    print("=" * 60)
    print("DRONE VS BIRD DATASET STATISTICS")
    print("=" * 60)

    print(f"\n📁 Data directory: {data_dir}")
    print(f"📁 Splits directory: {splits_dir}")

    print("\n" + "-" * 40)
    print("SPLIT SIZES")
    print("-" * 40)
    print(f"Train set:      {len(train_dataset):>6} images")
    print(f"Validation set: {len(val_dataset):>6} images")
    print(f"Test set:       {len(test_dataset):>6} images")
    print(f"Total:          {len(train_dataset) + len(val_dataset) + len(test_dataset):>6} images")

    print("\n" + "-" * 40)
    print("CLASS DISTRIBUTION")
    print("-" * 40)

    for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        label_counts = torch.bincount(torch.tensor(dataset.labels), minlength=2)
        drone_pct = 100 * label_counts[0].item() / len(dataset)
        bird_pct = 100 * label_counts[1].item() / len(dataset)
        print(
            f"{name:12} | Drone: {label_counts[0].item():>4} ({drone_pct:5.1f}%) | Bird: {label_counts[1].item():>4} ({bird_pct:5.1f}%)"
        )

    print("\n" + "-" * 40)
    print("IMAGE STATISTICS (sampled)")
    print("-" * 40)
    print(
        f"Width:  min={img_stats['min_width']:.0f}, max={img_stats['max_width']:.0f}, avg={img_stats['avg_width']:.1f}"
    )
    print(
        f"Height: min={img_stats['min_height']:.0f}, max={img_stats['max_height']:.0f}, avg={img_stats['avg_height']:.1f}"
    )


@app.command()
def dataset_statistics(
    data_dir: Path = typer.Option(Path("data"), help="Directory containing the image data"),
    splits_dir: Path = typer.Option(Path("data/splits"), help="Directory containing split files"),
    output_dir: Path = typer.Option(Path("reports/figures"), help="Directory to save figures"),
    markdown: bool = typer.Option(False, "--markdown", "-m", help="Output statistics in markdown format for CI"),
) -> None:
    """Compute and display dataset statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_dataset = DroneVsBirdDataset(data_dir, splits_dir / "train_files.txt")
    val_dataset = DroneVsBirdDataset(data_dir, splits_dir / "val_files.txt")
    test_dataset = DroneVsBirdDataset(data_dir, splits_dir / "test_files.txt")

    # Compute image statistics
    img_stats = get_image_stats(train_dataset)

    # Print statistics (markdown or console format)
    if markdown:
        print_markdown_report(train_dataset, val_dataset, test_dataset, img_stats)
    else:
        print_console_report(train_dataset, val_dataset, test_dataset, img_stats, data_dir, splits_dir)

        # Sample image info (console only)
        sample_img, sample_label = train_dataset[0]
        if hasattr(sample_img, "shape"):
            print(f"Transformed image shape: {tuple(sample_img.shape)}")
        print(f"Label type: {type(sample_label).__name__}")

    # Generate and save plots
    if not markdown:
        print("\n" + "-" * 40)
        print("SAVING FIGURES")
        print("-" * 40)

    # Sample images
    fig = show_sample_images(train_dataset, n_samples=16)
    fig.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not markdown:
        print(f"✓ Saved: {output_dir / 'sample_images.png'}")

    # Label distributions
    fig = plot_label_distribution(train_dataset.labels, "Train Label Distribution")
    fig.savefig(output_dir / "train_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not markdown:
        print(f"✓ Saved: {output_dir / 'train_label_distribution.png'}")

    fig = plot_label_distribution(val_dataset.labels, "Validation Label Distribution")
    fig.savefig(output_dir / "val_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not markdown:
        print(f"✓ Saved: {output_dir / 'val_label_distribution.png'}")

    fig = plot_label_distribution(test_dataset.labels, "Test Label Distribution")
    fig.savefig(output_dir / "test_label_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not markdown:
        print(f"✓ Saved: {output_dir / 'test_label_distribution.png'}")

    # Combined distribution
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#2ecc71", "#3498db"]

    for ax, (name, dataset) in zip(
        axes, [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]
    ):
        label_counts = torch.bincount(torch.tensor(dataset.labels), minlength=2)
        bars = ax.bar([LABEL_NAMES[0], LABEL_NAMES[1]], label_counts.numpy(), color=colors, edgecolor="black")
        for bar, count in zip(bars, label_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count.item()), ha="center", fontsize=10
            )
        ax.set_title(f"{name} (n={len(dataset)})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Label Distribution Across Splits", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "all_splits_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if not markdown:
        print(f"✓ Saved: {output_dir / 'all_splits_distribution.png'}")

    if not markdown:
        print("\n" + "=" * 60)
        print("Done!")


if __name__ == "__main__":
    app()
