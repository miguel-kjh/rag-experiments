from datasets import load_dataset
import argparse
import os

from utils import FOLDER_RAW

def download_dataset(dataset_name, split=None):
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    name_dataset = dataset_name.split("/")[-1]
    folder_path = os.path.join(FOLDER_RAW, name_dataset)
    dataset.save_to_disk(folder_path)
    print(f"Dataset saved to {folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face using the datasets library.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (e.g., 'imdb', 'squad')")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to download (e.g., 'train', 'test')")
    args = parser.parse_args()

    download_dataset(args.dataset_name, split=args.split)