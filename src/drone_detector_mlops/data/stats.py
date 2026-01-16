"""Dataset statistics and visualization module."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer

from drone_detector_mlops.data.data import DroneVsBirdDataset

app = typer.Typer()

LABEL_NAMES = {0: "drone", 1: "bird"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def count_images_in_directory(data_dir: Path) -> dict[str, int]:
    """Count image files directly in the data directories."""
    counts = {"drone": 0, "bird": 0}
    for class_name in counts:
        class_dir = data_dir / class_name
        if class_dir.exists():
            counts[class_name] = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    return counts


def find_new_images_in_directory(before_dir: Path, after_dir: Path) -> dict[str, list[str]]:
    """Find image files that exist in 'after' but not in 'before'."""
    new_images = {"drone": [], "bird": []}
    for class_name in new_images:
        before_class_dir = before_dir / class_name
        after_class_dir = after_dir / class_name

        before_files = set()
        if before_class_dir.exists():
            before_files = {f.name for f in before_class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}

        if after_class_dir.exists():
            for f in after_class_dir.iterdir():
                if f.suffix.lower() in IMAGE_EXTENSIONS and f.name not in before_files:
                    new_images[class_name].append(f.name)

    return new_images


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


def get_all_files_from_splits(splits_dir: Path) -> set[str]:
    """Get all file paths from all split files."""
    all_files = set()
    for split_file in ["train_files.txt", "val_files.txt", "test_files.txt"]:
        split_path = splits_dir / split_file
        if split_path.exists():
            all_files.update(split_path.read_text().strip().split("\n"))
    return all_files


def find_new_files(before_splits_dir: Path, after_splits_dir: Path) -> list[str]:
    """Find files that exist in 'after' but not in 'before'."""
    before_files = get_all_files_from_splits(before_splits_dir)
    after_files = get_all_files_from_splits(after_splits_dir)
    return sorted(after_files - before_files)


def show_before_and_new_images(
    before_dataset: DroneVsBirdDataset,
    after_dataset: DroneVsBirdDataset,
    new_files: list[str],
    after_data_dir: Path,
    n_before: int = 8,
    n_new: int = 8,
) -> plt.Figure:
    """Create a grid showing 'before' sample images and new images side by side.

    Args:
        before_dataset: Dataset from before state (for random sample images)
        after_dataset: Dataset from after state (used for image loading utility)
        new_files: List of new file paths like "bird/image.jpeg"
        after_data_dir: Directory containing the after data (to load new images directly)
        n_before: Number of before samples to show
        n_new: Max number of new images to show
    """
    from PIL import Image

    # Get before samples
    before_indices = torch.randperm(len(before_dataset))[:n_before]

    # Load new images directly from directory (not from split-based dataset)
    new_images_to_show = new_files[:n_new]

    n_before_actual = len(before_indices)
    n_new_actual = len(new_images_to_show)
    n_cols = 4

    # Calculate rows needed for each section
    before_rows = (n_before_actual + n_cols - 1) // n_cols
    new_rows = (n_new_actual + n_cols - 1) // n_cols if n_new_actual > 0 else 1

    fig, axes = plt.subplots(before_rows + new_rows, n_cols, figsize=(12, 3 * (before_rows + new_rows)))
    if before_rows + new_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.reshape(-1, n_cols)

    # Plot before images
    for row in range(before_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            idx_pos = row * n_cols + col
            if idx_pos < n_before_actual:
                data_idx = before_indices[idx_pos]
                img_path = before_dataset._build_image_path(data_idx.item())
                image = before_dataset._open_image(img_path)
                label = before_dataset.labels[data_idx.item()]
                ax.imshow(image)
                title = f"{LABEL_NAMES[label]}"
                if row == 0 and col == 0:
                    title = f"[BEFORE] {title}"
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    # Plot new images (loaded directly from directory)
    if n_new_actual > 0:
        for row in range(new_rows):
            for col in range(n_cols):
                ax = axes[before_rows + row, col]
                idx_pos = row * n_cols + col
                if idx_pos < n_new_actual:
                    rel_path = new_images_to_show[idx_pos]
                    img_path = after_data_dir / rel_path
                    # Determine class from path
                    class_name = rel_path.split("/")[0]
                    try:
                        image = Image.open(img_path).convert("RGB")
                        ax.imshow(image)
                        title = f"{class_name}"
                        if row == 0 and col == 0:
                            title = f"[NEW] {title}"
                        ax.set_title(title, fontsize=10, color="green")
                    except Exception:
                        ax.text(0.5, 0.5, "Error loading", ha="center", va="center")
                ax.axis("off")
    else:
        # No new images - show placeholder
        ax = axes[before_rows, 0]
        ax.text(0.5, 0.5, "No new images", ha="center", va="center", fontsize=14)
        ax.axis("off")
        for col in range(1, n_cols):
            axes[before_rows, col].axis("off")

    fig.tight_layout()
    return fig


def plot_comparison_distribution(
    before_datasets: dict[str, DroneVsBirdDataset],
    after_datasets: dict[str, DroneVsBirdDataset],
) -> plt.Figure:
    """Plot before/after label distribution comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors_before = ["#a8d5a2", "#a8c8e8"]  # lighter green/blue for before
    colors_after = ["#2ecc71", "#3498db"]  # darker green/blue for after

    bar_width = 0.35
    x = [0, 1]

    for ax, split_name in zip(axes, ["Train", "Validation", "Test"]):
        before_ds = before_datasets[split_name]
        after_ds = after_datasets[split_name]

        before_counts = torch.bincount(torch.tensor(before_ds.labels), minlength=2).numpy()
        after_counts = torch.bincount(torch.tensor(after_ds.labels), minlength=2).numpy()

        # Plot grouped bars
        x_before = [i - bar_width / 2 for i in x]
        x_after = [i + bar_width / 2 for i in x]

        bars_before = ax.bar(x_before, before_counts, bar_width, label="Before", color=colors_before, edgecolor="black")
        bars_after = ax.bar(x_after, after_counts, bar_width, label="After", color=colors_after, edgecolor="black")

        # Add count labels
        for bar, count in zip(bars_before, before_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(count), ha="center", fontsize=9)
        for bar, count in zip(bars_after, after_counts):
            diff = count - before_counts[list(bars_after).index(bar)]
            diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else ""
            label = f"{count}" + (f" ({diff_str})" if diff_str else "")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, label, ha="center", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels([LABEL_NAMES[0], LABEL_NAMES[1]])
        ax.set_title(
            f"{split_name}\n(before: {len(before_ds)}, after: {len(after_ds)})", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Label Distribution: Before vs After", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def print_comparison_markdown_report(
    before_datasets: dict[str, DroneVsBirdDataset],
    after_datasets: dict[str, DroneVsBirdDataset],
    before_img_stats: dict,
    after_img_stats: dict,
    new_files: list[str],
    before_dir_counts: dict[str, int],
    after_dir_counts: dict[str, int],
    new_images_in_dirs: dict[str, list[str]],
) -> None:
    """Print comparison dataset statistics in markdown format."""
    print("# 📊 Drone vs Bird Dataset Changes\n")

    # Summary of raw data changes (directory level)
    before_dir_total = sum(before_dir_counts.values())
    after_dir_total = sum(after_dir_counts.values())
    dir_diff = after_dir_total - before_dir_total
    dir_diff_str = f"+{dir_diff}" if dir_diff > 0 else str(dir_diff)
    total_new_images = sum(len(imgs) for imgs in new_images_in_dirs.values())

    print("## Raw Data Changes\n")
    print(f"**Total images in data directories:** {before_dir_total} → {after_dir_total} ({dir_diff_str})\n")
    print("| Class | Before | After | Diff |")
    print("|-------|--------|-------|------|")
    for class_name in ["drone", "bird"]:
        before_count = before_dir_counts.get(class_name, 0)
        after_count = after_dir_counts.get(class_name, 0)
        diff = after_count - before_count
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
        print(f"| {class_name} | {before_count} | {after_count} | {diff_str} |")
    print()

    # List new images
    if total_new_images > 0:
        print(f"### New Images Added ({total_new_images} total)\n")
        for class_name in ["drone", "bird"]:
            new_imgs = new_images_in_dirs.get(class_name, [])
            if new_imgs:
                print(f"**{class_name}:** {len(new_imgs)} new images")
                for img in new_imgs[:5]:
                    print(f"- `{class_name}/{img}`")
                if len(new_imgs) > 5:
                    print(f"- ... and {len(new_imgs) - 5} more")
        print()

    # Summary of split-based changes
    before_total = sum(len(ds) for ds in before_datasets.values())
    after_total = sum(len(ds) for ds in after_datasets.values())
    diff_total = after_total - before_total
    diff_str = f"+{diff_total}" if diff_total > 0 else str(diff_total)

    if diff_total == 0 and dir_diff > 0:
        print("> ⚠️ **Note:** New images were added to data directories but split files have not been updated.\n")

    # Split sizes comparison table
    print("## Split Sizes\n")
    print("| Split | Before | After | Diff |")
    print("|-------|--------|-------|------|")
    for split_name in ["Train", "Validation", "Test"]:
        before_len = len(before_datasets[split_name])
        after_len = len(after_datasets[split_name])
        diff = after_len - before_len
        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
        print(f"| {split_name} | {before_len} | {after_len} | {diff_str} |")

    before_total = sum(len(ds) for ds in before_datasets.values())
    after_total = sum(len(ds) for ds in after_datasets.values())
    diff_total = after_total - before_total
    diff_str = f"+{diff_total}" if diff_total > 0 else str(diff_total) if diff_total < 0 else "0"
    print(f"| **Total** | **{before_total}** | **{after_total}** | **{diff_str}** |")
    print()

    # Class distribution comparison table
    print("## Class Distribution\n")
    print("| Split | Class | Before | After | Diff |")
    print("|-------|-------|--------|-------|------|")

    for split_name in ["Train", "Validation", "Test"]:
        before_ds = before_datasets[split_name]
        after_ds = after_datasets[split_name]
        before_counts = torch.bincount(torch.tensor(before_ds.labels), minlength=2)
        after_counts = torch.bincount(torch.tensor(after_ds.labels), minlength=2)

        for class_idx, class_name in LABEL_NAMES.items():
            before_count = before_counts[class_idx].item()
            after_count = after_counts[class_idx].item()
            diff = after_count - before_count
            diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
            print(f"| {split_name} | {class_name} | {before_count} | {after_count} | {diff_str} |")
    print()

    # New files list (first few)
    if new_files:
        print("## New Files Added\n")
        print(f"Total new files: **{len(new_files)}**\n")
        if len(new_files) <= 20:
            for f in new_files:
                print(f"- `{f}`")
        else:
            for f in new_files[:10]:
                print(f"- `{f}`")
            print(f"- ... and {len(new_files) - 10} more")
        print()


@app.command()
def dataset_statistics(
    data_dir: Path = typer.Option(Path("data"), help="Directory containing the image data"),
    splits_dir: Path = typer.Option(Path("data/splits"), help="Directory containing split files"),
    output_dir: Path = typer.Option(Path("reports/figures"), help="Directory to save figures"),
    markdown: bool = typer.Option(False, "--markdown", "-m", help="Output statistics in markdown format for CI"),
    before_data_dir: Path = typer.Option(
        None, "--before-data-dir", help="Directory containing 'before' image data for comparison"
    ),
    before_splits_dir: Path = typer.Option(
        None, "--before-splits-dir", help="Directory containing 'before' split files for comparison"
    ),
) -> None:
    """Compute and display dataset statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we're in comparison mode
    comparison_mode = before_data_dir is not None and before_splits_dir is not None

    # Load current (after) datasets
    train_dataset = DroneVsBirdDataset(data_dir, splits_dir / "train_files.txt")
    val_dataset = DroneVsBirdDataset(data_dir, splits_dir / "val_files.txt")
    test_dataset = DroneVsBirdDataset(data_dir, splits_dir / "test_files.txt")
    after_datasets = {"Train": train_dataset, "Validation": val_dataset, "Test": test_dataset}

    # Compute image statistics
    img_stats = get_image_stats(train_dataset)

    if comparison_mode:
        # Load before datasets
        before_train = DroneVsBirdDataset(before_data_dir, before_splits_dir / "train_files.txt")
        before_val = DroneVsBirdDataset(before_data_dir, before_splits_dir / "val_files.txt")
        before_test = DroneVsBirdDataset(before_data_dir, before_splits_dir / "test_files.txt")
        before_datasets = {"Train": before_train, "Validation": before_val, "Test": before_test}

        before_img_stats = get_image_stats(before_train)
        new_files = find_new_files(before_splits_dir, splits_dir)

        # Count images directly in directories (independent of splits)
        before_dir_counts = count_images_in_directory(before_data_dir)
        after_dir_counts = count_images_in_directory(data_dir)
        new_images_in_dirs = find_new_images_in_directory(before_data_dir, data_dir)

        # Print comparison report
        if markdown:
            print_comparison_markdown_report(
                before_datasets,
                after_datasets,
                before_img_stats,
                img_stats,
                new_files,
                before_dir_counts,
                after_dir_counts,
                new_images_in_dirs,
            )
        else:
            print("Comparison mode - console output not fully implemented")
            print(f"Before data: {before_data_dir}")
            print(f"After data: {data_dir}")
            print(f"New files in splits: {len(new_files)}")
            print(f"New images in directories: {sum(len(imgs) for imgs in new_images_in_dirs.values())}")

        # Generate comparison figures
        if not markdown:
            print("\n" + "-" * 40)
            print("SAVING FIGURES")
            print("-" * 40)

        # Before/new sample images - use directory-level new images
        all_new_files = [f"{cls}/{img}" for cls, imgs in new_images_in_dirs.items() for img in imgs]
        fig = show_before_and_new_images(before_train, train_dataset, all_new_files, data_dir, n_before=8, n_new=8)
        fig.savefig(output_dir / "sample_images.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        if not markdown:
            print(f"✓ Saved: {output_dir / 'sample_images.png'}")

        # Comparison distribution
        fig = plot_comparison_distribution(before_datasets, after_datasets)
        fig.savefig(output_dir / "all_splits_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        if not markdown:
            print(f"✓ Saved: {output_dir / 'all_splits_distribution.png'}")

    else:
        # Normal mode (no comparison)
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
