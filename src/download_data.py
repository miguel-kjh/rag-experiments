from datasets import load_dataset
import argparse
import os

from utils import (
    FOLDER_RAW,
    RANGBENCH_SUBSETS,
)

def download_dataset(dataset_name, split=None):
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    name_dataset = dataset_name.split("/")[-1]
    folder_path = os.path.join(FOLDER_RAW, name_dataset)
    dataset.save_to_disk(folder_path)
    print(f"Dataset saved to {folder_path}")

def download_ragbench():
    for subset in RANGBENCH_SUBSETS:
        print(f"Downloading RAG-Bench subset: {subset}")
        dataset = load_dataset("rungalileo/ragbench", subset)
        folder_path = os.path.join(FOLDER_RAW, f"ragbench-{subset}")
        dataset.save_to_disk(folder_path)
        print(f"RAG-Bench subset saved to {folder_path}")

DATASETS_TO_DOWNLOAD = {
    "ragbench": download_ragbench,
}


def main(args: argparse.Namespace):
    if args.dataset_name in DATASETS_TO_DOWNLOAD:
        DATASETS_TO_DOWNLOAD[args.dataset_name]()
    else:
        download_dataset(args.dataset_name, split=args.split)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face using the datasets library.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset (e.g., 'imdb', 'squad')")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to download (e.g., 'train', 'test')")
    args = parser.parse_args()

    main(args)