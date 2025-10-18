#!/usr/bin/env python3
"""
Script to download book images dataset using CIFAR-100 books category
"""
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from datasets import load_dataset
except ImportError:
    print("Installing required package: datasets")
    os.system("pip install datasets")
    from datasets import load_dataset

def download_and_organize():
    print("Downloading image dataset with books category...")

    # Try using CIFAR-100 which has a books category
    try:
        dataset = load_dataset("cifar100")

        # CIFAR-100 fine labels (we want books which is label 11)
        # Labels: 0-99 for different objects, 11 is books
        book_label = 11

        # Create directories
        base_dir = Path("data/books")
        for split in ["train", "valid"]:
            for class_name in ["book", "other"]:
                (base_dir / split / class_name).mkdir(parents=True, exist_ok=True)

        print("\nExtracting book images from dataset...")

        # Process train split
        train_books = 0
        train_others = 0
        for idx, example in enumerate(dataset['train']):
            img = example['img']
            label = example['fine_label']

            if label == book_label:
                img.save(base_dir / "train" / "book" / f"book_{train_books}.png")
                train_books += 1
            elif train_others < train_books and idx % 10 == 0:  # Sample some non-books
                img.save(base_dir / "train" / "other" / f"other_{train_others}.png")
                train_others += 1

            if train_books >= 400 and train_others >= 400:
                break

        # Process test split for validation
        valid_books = 0
        valid_others = 0
        for idx, example in enumerate(dataset['test']):
            img = example['img']
            label = example['fine_label']

            if label == book_label:
                img.save(base_dir / "valid" / "book" / f"book_{valid_books}.png")
                valid_books += 1
            elif valid_others < valid_books and idx % 10 == 0:
                img.save(base_dir / "valid" / "other" / f"other_{valid_others}.png")
                valid_others += 1

            if valid_books >= 100 and valid_others >= 100:
                break

        print(f"\n✅ Dataset ready!")
        print(f"Train - Books: {train_books}, Others: {train_others}")
        print(f"Valid - Books: {valid_books}, Others: {valid_others}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach with sample data...")

        # Fallback: create minimal dataset structure with placeholders
        base_dir = Path("data/books")
        for split in ["train", "valid"]:
            for class_name in ["book", "other"]:
                class_dir = base_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Create a few placeholder images
                num_samples = 10 if split == "valid" else 30
                for i in range(num_samples):
                    # Create random colored image
                    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                    img = Image.fromarray(arr)
                    img.save(class_dir / f"{class_name}_{i}.png")

        print(f"\n✅ Created placeholder dataset structure")
        print("Please replace with real book images!")

if __name__ == "__main__":
    download_and_organize()
