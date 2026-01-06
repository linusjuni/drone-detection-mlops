from pathlib import Path
from sklearn.model_selection import train_test_split

from drone_detector_mlops.utils.settings import settings

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        for img_path in class_dir.glob("*.[jJ][pP]*[gG]"):
            image_paths.append(img_path.relative_to(data_dir))
            labels.append(class_name)

    # Stratified split: first split off test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_RATIO, stratify=labels, random_state=settings.RANDOM_SEED
    )

    # Split remaining into train and val
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_ratio_adjusted,
        stratify=train_val_labels,
        random_state=settings.RANDOM_SEED,
    )

    # Save splits
    for split_name, paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
        split_file = output_dir / f"{split_name}_files.txt"
        with open(split_file, "w") as f:
            f.write("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    data_dir = Path("data")
    output_dir = data_dir / "splits"
    create_splits(data_dir, output_dir)
