import argparse
import os
from datasets import load_from_disk
from langchain_core.documents import Document
import pickle

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer
from ingestion import Ingestion
from utils import (
    FOLDER_DB,
    FOLDER_PROCESSED,
    RAGBENCH_SUBSETS
)


def create_db_for_ragbench(model_name: str, max_length: int, batch_size: int = 16, using_chunking: bool = False):
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )
    for subset in RAGBENCH_SUBSETS:
        print(f"Creating DB for RAG-Bench subset: {subset}")
        folder_path = os.path.join(FOLDER_DB, f"ragbench-{subset}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        unique_documents = pickle.load(open(os.path.join(FOLDER_PROCESSED, f"ragbench-{subset}", "unique_documents.pkl"), "rb"))
        print(f"Loaded {len(unique_documents)} unique documents from {os.path.join(FOLDER_PROCESSED, f'ragbench-{subset}', 'unique_documents.pkl')}")
        documents = [
            Document(
                page_content=doc,
                metadata={
                    "id": id,
                }
            ) for id, doc in enumerate(unique_documents)
        ]
        ingestion = Ingestion(documents=documents, embeddings=model, batch_size=batch_size)
        faiss_index = ingestion.ingest()
        name_db = os.path.join(
            folder_path,
            f"ragbench-{subset}_embeddings_{model_name.replace('/', '_')}"
        )
        print(f"Saving FAISS index to {name_db}")
        faiss_index.save_local(name_db)

def create_db_for_parliament(model_name: str, max_length: int, batch_size: int = 16, using_chunking: bool = False):
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"}
    )
    
    text_splitter = TokenTextSplitter(
        chunk_size=max_length, 
        chunk_overlap=0
    )

    print("Creating DB for Parliament dataset") 
    folder_path = os.path.join(FOLDER_DB, "parliament_db")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    db_documents_data = load_from_disk(os.path.join(FOLDER_PROCESSED, "parliament_all_docs"))
    documents = []

    for doc in db_documents_data["all"]:
        chunks = text_splitter.split_text(doc["text"])
        if using_chunking:
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "id": doc["PK"],
                    }
                ))
        else:
            documents.append(Document(
                page_content=chunks[0],
                metadata={
                    "id": doc["PK"],
                }
            ))

    
    ingestion = Ingestion(documents=documents, embeddings=model, batch_size=batch_size)
    faiss_index = ingestion.ingest()
    using_chunking_str = "_chunked" if using_chunking else ""
    max_length_str = f"_max_length-{max_length}"
    name_db = os.path.join(
        folder_path,
        f"parliament_all_docs_embeddings_{model_name.replace('/', '_')}{using_chunking_str}{max_length_str}"
    )
    print(f"Saving FAISS index to {name_db}")
    faiss_index.save_local(name_db)


DATASETS = {
    "ragbench": create_db_for_ragbench,
    "parliament": create_db_for_parliament,
}

def main(args: argparse.Namespace):
    if args.dataset in DATASETS:
        DATASETS[args.dataset](args.model_name, args.max_length, args.batch_size, args.using_chunking)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to local dataset (load_from_disk) or Hub dataset ID (load_dataset)")
    parser.add_argument("--model_name", required=True, help="Name of the model to use for embeddings")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length for the model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--using_chunking", action="store_true", default=False, help="Whether to use chunking for long documents")
    args = parser.parse_args() 

    main(args)