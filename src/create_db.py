import argparse
import os
from datasets import load_from_disk
from langchain_core.documents import Document
from datasets import concatenate_datasets, DatasetDict

from embeddings_models import SentenceTransformerEmbeddings
from ingestion import Ingestion
from utils import FOLDER_DB



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to local dataset (load_from_disk) or Hub dataset ID (load_dataset)")
    parser.add_argument("--model_name", required=True, help="Name of the model to use for embeddings")
    args = parser.parse_args()

    dataset = load_from_disk(args.source)
    
    # si hay varias particiones, las concatenamos
    if len(dataset.keys()) > 1:
        dataset = concatenate_datasets(list(dataset.values()))
        dataset = DatasetDict({"train": dataset})  # lo guardamos como train    

    model = SentenceTransformerEmbeddings(model=args.model_name)

    # TODO: chunking

    # tranform dataset to list of Documents
    documents = [
        Document(
            page_content=entry['answer'],
            metadata={
                "id": id,
                "question": entry['question']
            }
        ) for id, entry in enumerate(dataset['train'])
    ]

    # create ingestion object
    ingestion = Ingestion(documents=documents, embeddings=model)
    faiss_index = ingestion.ingest()

    # save faiss index
    name_db = os.path.join(
        FOLDER_DB,
        f"{os.path.basename(args.source)}_embeddings_{args.model_name.replace('/', '_')}"
    )

    print(f"Saving FAISS index to {name_db}")
    faiss_index.save_local(name_db)

if __name__ == "__main__":
    main()