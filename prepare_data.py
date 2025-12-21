"""
Data preparation script for raccoon training dataset.

This script helps you prepare a dataset of raccoon images for training.
You'll need to source your own raccoon images.

Options:
1. Download from datasets like Kaggle or HuggingFace
2. Use DuckDuckGo image search to scrape images
3. Use your own collection of raccoon images

Usage:
    # Option 1: Prepare existing images
    python prepare_data.py --source_dir ./raw_raccoons --output_dir ./raccoon_data

    # Option 2: Download from DuckDuckGo (requires duckduckgo_search)
    python prepare_data.py --download --num_images 500 --output_dir ./raccoon_data
"""

import argparse
import os
from pathlib import Path
import hashlib

from PIL import Image
from tqdm import tqdm


def download_images(output_dir: str, num_images: int = 500):
    """
    Download raccoon images using DuckDuckGo search.
    Requires: pip install ddgs
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("Please install ddgs: pip install ddgs")
            return

    import requests
    import time
    import random

    output_path = Path(output_dir) / "raccoons"
    output_path.mkdir(parents=True, exist_ok=True)

    search_terms = [
        "raccoon face",
        "raccoon portrait",
        "cute raccoon",
        "raccoon close up",
        "baby raccoon",
        "raccoon looking at camera",
    ]

    downloaded = 0
    for term in search_terms:
        if downloaded >= num_images:
            break

        print(f"Searching for: {term}")

        # Add delay between searches to avoid rate limiting
        time.sleep(random.uniform(2, 5))

        max_retries = 3
        for retry in range(max_retries):
            try:
                ddgs = DDGS()
                results = list(ddgs.images(term, max_results=num_images // len(search_terms) + 50))
                break
            except Exception as e:
                if "Ratelimit" in str(e) and retry < max_retries - 1:
                    wait_time = (retry + 1) * 10
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"Error searching '{term}': {e}")
                    results = []
                    break

        for result in tqdm(results, desc=f"Downloading '{term}'"):
            if downloaded >= num_images:
                break

            try:
                url = result["image"]
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Use hash of URL for filename to avoid duplicates
                    filename = hashlib.md5(url.encode()).hexdigest()[:16] + ".jpg"
                    filepath = output_path / filename

                    if not filepath.exists():
                        with open(filepath, "wb") as f:
                            f.write(response.content)

                        # Verify it's a valid image
                        try:
                            img = Image.open(filepath)
                            img.verify()
                            downloaded += 1
                        except Exception:
                            filepath.unlink()

                # Small delay between downloads
                time.sleep(random.uniform(0.1, 0.3))
            except Exception as e:
                continue

    print(f"Downloaded {downloaded} images to {output_path}")


def prepare_images(source_dir: str, output_dir: str, img_size: int = 64):
    """
    Prepare images for training:
    - Resize to target size
    - Convert to RGB
    - Filter out bad images
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir) / "raccoons"
    output_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    # Find all images
    image_files = []
    for ext in valid_extensions:
        image_files.extend(source_path.glob(f"**/*{ext}"))
        image_files.extend(source_path.glob(f"**/*{ext.upper()}"))

    print(f"Found {len(image_files)} images")

    processed = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img = Image.open(img_path)

            # Convert to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Get dimensions
            w, h = img.size

            # Skip very small images
            if w < 64 or h < 64:
                continue

            # Center crop to square
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))

            # Resize
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)

            # Save
            output_file = output_path / f"raccoon_{processed:05d}.png"
            img.save(output_file)
            processed += 1

        except Exception as e:
            continue

    print(f"Processed {processed} images to {output_path}")
    print(f"\nDataset ready! You can now train with:")
    print(f"  python train.py --data_dir {output_dir} --epochs 100")


def verify_dataset(data_dir: str):
    """Verify the dataset is ready for training."""
    data_path = Path(data_dir)

    # Check for subdirectory (required by ImageFolder)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"Error: {data_dir} needs a subdirectory containing images")
        print("Expected structure:")
        print(f"  {data_dir}/")
        print(f"  └── raccoons/")
        print("      ├── image1.png")
        print("      ├── image2.png")
        print("      └── ...")
        return False

    # Count images
    total_images = 0
    for subdir in subdirs:
        images = list(subdir.glob("*.png")) + list(subdir.glob("*.jpg"))
        total_images += len(images)
        print(f"  {subdir.name}/: {len(images)} images")

    print(f"\nTotal: {total_images} images")

    if total_images < 100:
        print("Warning: Less than 100 images. Consider adding more for better results.")
    elif total_images < 500:
        print("Tip: 500+ images recommended for good quality.")
    else:
        print("Dataset size looks good!")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare raccoon training dataset")
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Directory containing raw raccoon images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./raccoon_data",
        help="Output directory for processed images",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download images from DuckDuckGo",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of images to download",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=64,
        help="Target image size",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing dataset",
    )

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
    elif args.download:
        download_images(args.output_dir, args.num_images)
        prepare_images(args.output_dir, args.output_dir, args.img_size)
    elif args.source_dir:
        prepare_images(args.source_dir, args.output_dir, args.img_size)
    else:
        print("Usage:")
        print("  Download images:  python prepare_data.py --download --num_images 500")
        print("  Prepare existing: python prepare_data.py --source_dir ./raw_images")
        print("  Verify dataset:   python prepare_data.py --verify")
