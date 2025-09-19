from datasets import load_dataset
import argparse
import pickle
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
    columns_to_keep = ["id", "question", "documents", "response"]
    for subset in RANGBENCH_SUBSETS:
        print(f"Downloading RAG-Bench subset: {subset}")
        dataset = load_dataset("rungalileo/ragbench", subset)
        documents = []
        for split in dataset:
            for docs in dataset[split]["documents"]:
                for doc in docs:
                    documents.append(doc)
        unique_documents = list(set(documents))
        unique_documents.sort()
        print(f"Number of unique documents in {subset}: {len(unique_documents)}")
        document_idx_map = {doc: idx for idx, doc in enumerate(unique_documents)}
        # Keep only specified columns in each split
        for split in dataset.keys():
            dataset[split] = dataset[split].remove_columns(
                [col for col in dataset[split].column_names if col not in columns_to_keep]
            )
            dataset[split] = dataset[split].add_column(
                "document_ids",
                [[document_idx_map[doc] for doc in docs] for docs in dataset[split]["documents"]]
            )
        print(f"Columns in {subset} after processing: {dataset['train'].column_names}")
        # Save the entire dataset (all splits) to disk
        folder_path = os.path.join(FOLDER_RAW, f"ragbench-{subset}")
        dataset.save_to_disk(folder_path)
        print(f"RAG-Bench subset saved to {folder_path}")
        # save unique documents to a pkl file
        with open(os.path.join(folder_path, "unique_documents.pkl"), "wb") as f:
            pickle.dump(unique_documents, f)
        print(f"Unique documents saved to {os.path.join(folder_path, 'unique_documents.pkl')}")

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