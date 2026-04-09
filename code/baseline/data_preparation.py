import random
from pathlib import Path
from collections import defaultdict
import json

from config import (
    DATA_DIR, 
    TRAIN_SPLIT, 
    VAL_SPLIT, 
    TEST_SPLIT, 
    RANDOM_SEED,
    RESULTS_DIR
)


def get_all_images(data_dir):
    images_by_class = defaultdict(list)
    
    for class_folder in sorted(data_dir.iterdir()):
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        
        image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
        for img_file in class_folder.iterdir():
            if img_file.suffix in image_extensions:
                images_by_class[class_name].append(str(img_file))
    
    return images_by_class


def split_data(images_by_class, train_split, val_split, test_split, random_seed):
    random.seed(random_seed)
    
    train_files = []
    val_files = []
    test_files = []
    
    class_to_idx = {class_name: idx for idx, class_name in enumerate(sorted(images_by_class.keys()))}
    
    for class_name, image_paths in sorted(images_by_class.items()):
        random.shuffle(image_paths)
        
        n_total = len(image_paths)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_paths = image_paths[:n_train]
        val_paths = image_paths[n_train:n_train + n_val]
        test_paths = image_paths[n_train + n_val:]
        
        label = class_to_idx[class_name]
        
        train_files.extend([(path, label) for path in train_paths])
        val_files.extend([(path, label) for path in val_paths])
        test_files.extend([(path, label) for path in test_paths])
        
        print(f"{class_name}: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    
    return train_files, val_files, test_files, class_to_idx


def save_split_info(train_files, val_files, test_files, class_to_idx, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
    
    with open(output_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "idx_to_class.json", "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2, ensure_ascii=False)
    
    splits = {
        "train": [{"path": path, "label": label} for path, label in train_files],
        "val": [{"path": path, "label": label} for path, label in val_files],
        "test": [{"path": path, "label": label} for path, label in test_files]
    }
    
    with open(output_dir / "data_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)


def main():
    images_by_class = get_all_images(DATA_DIR)
    
    train_files, val_files, test_files, class_to_idx = split_data(
        images_by_class, 
        TRAIN_SPLIT, 
        VAL_SPLIT, 
        TEST_SPLIT, 
        RANDOM_SEED
    )
    
    save_split_info(train_files, val_files, test_files, class_to_idx, RESULTS_DIR)


if __name__ == "__main__":
    main()
