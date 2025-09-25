import argparse
import os
from datasets import load_from_disk
from langchain_core.documents import Document
from datasets import concatenate_datasets, DatasetDict
import pickle

from embeddings_models import SentenceTransformerEmbeddings
from ingestion import Ingestion
from utils import (
    FOLDER_DB,
    FOLDER_RAW,
    FOLDER_PROCESSED,
    RAGBENCH_SUBSETS
)


def create_db_for_ragbench(model_name: str):
    model = SentenceTransformerEmbeddings(model=model_name)
    for subset in RAGBENCH_SUBSETS:
        print(f"Creating DB for RAG-Bench subset: {subset}")
        folder_path = os.path.join(FOLDER_DB, f"ragbench-{subset}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        unique_documents = pickle.load(open(os.path.join(FOLDER_RAW, f"ragbench-{subset}", "unique_documents.pkl"), "rb"))
        print(f"Loaded {len(unique_documents)} unique documents from {os.path.join(FOLDER_RAW, f'ragbench-{subset}', 'unique_documents.pkl')}")
        documents = [
            Document(
                page_content=doc,
                metadata={
                    "id": id,
                }
            ) for id, doc in enumerate(unique_documents)
        ]
        ingestion = Ingestion(documents=documents, embeddings=model)
        faiss_index = ingestion.ingest()
        name_db = os.path.join(
            folder_path,
            f"ragbench-{subset}_embeddings_all-mpnet-base-v2"
        )
        print(f"Saving FAISS index to {name_db}")
        faiss_index.save_local(name_db)

def create_db_for_parliament(model_name: str):
    model = SentenceTransformerEmbeddings(model=model_name)
    print("Creating DB for Parliament dataset")
    folder_path = os.path.join(FOLDER_DB, "parliament_db")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    db_documents_data = load_from_disk(os.path.join(FOLDER_PROCESSED, "parliament_all_docs"))
    documents = [
        Document(
            page_content=doc["text"],
            metadata={
                "id": doc["PK"],
            }
        ) for doc in db_documents_data
    ]
    ingestion = Ingestion(documents=documents, embeddings=model)
    faiss_index = ingestion.ingest()
    name_db = os.path.join(
        folder_path,
        f"parliament_all_docs_embeddings_{model_name.replace('/', '_')}"
    )
    print(f"Saving FAISS index to {name_db}")
    faiss_index.save_local(name_db)


DATASETS = {
    "ragbench": create_db_for_ragbench,
    "parliament": create_db_for_parliament,
}

def main(args: argparse.Namespace):
    if args.dataset in DATASETS:
        DATASETS[args.dataset](args.model_name)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to local dataset (load_from_disk) or Hub dataset ID (load_dataset)")
    parser.add_argument("--model_name", required=True, help="Name of the model to use for embeddings")
    args = parser.parse_args() 

    main(args)