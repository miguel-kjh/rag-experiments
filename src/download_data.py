from datasets import (
    load_dataset, 
    load_from_disk,
    concatenate_datasets,
    DatasetDict,
    Dataset
)
import argparse
import pickle
import os

from utils import (
    FOLDER_RAW,
    FOLDER_PROCESSED,
    RAGBENCH_SUBSETS,
    SEED,
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
    for subset in RAGBENCH_SUBSETS:
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
        folder_path = os.path.join(FOLDER_PROCESSED, f"ragbench-{subset}")
        dataset.save_to_disk(folder_path)
        print(f"RAG-Bench subset saved to {folder_path}")
        # save unique documents to a pkl file
        with open(os.path.join(folder_path, "unique_documents.pkl"), "wb") as f:
            pickle.dump(unique_documents, f)
        print(f"Unique documents saved to {os.path.join(folder_path, 'unique_documents.pkl')}")

def download_clapnq():
    print("Downloading CLAP-NQ dataset")
    columns_of_final_dataset = ["id", "question", "documents", "response"]
    
    dataset = load_dataset("PrimeQA/clapnq")

    dataset = dataset.rename_columns({
        "input": "question",
        "passages": "documents",
        "output": "response"
    })

    def process(example):
        example["documents"] = [doc["text"] for doc in example["documents"]]
        example["response"] = example["response"][0]["answer"]
        return example

    dataset = dataset.map(process)

    # Reordenar columnas
    dataset = dataset.map(
        lambda x: {col: x[col] for col in columns_of_final_dataset}
    )

    dataset = DatasetDict({
        "train": dataset["train"],
        "validation": dataset["validation"],
        "test": dataset["validation"]
    })

    db_documents = []
    for split in dataset:
        for docs in dataset[split]["documents"]:
            for doc in docs:
                db_documents.append(doc)
    unique_documents = list(set(db_documents))
    unique_documents.sort()
    print(f"Number of unique documents in CLAP-NQ: {len(unique_documents)}")
    document_idx_map = {doc: idx for idx, doc in enumerate(unique_documents)}
    for split in dataset.keys():
        dataset[split] = dataset[split].add_column(
            "document_ids",
            [[document_idx_map[doc] for doc in docs] for docs in dataset[split]["documents"]]
        )
    #save dataset and unique documents
    folder_path = os.path.join(FOLDER_PROCESSED, "clapnq")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, "unique_documents.pkl"), "wb") as f:
        pickle.dump(unique_documents, f)
    dataset.save_to_disk(folder_path)
    print(f"CLAP-NQ dataset saved to {folder_path}")

def download_parliament():
    columns_of_final_dataset = ["id", "question", "documents", "response"]
    # pre-trained dataset with train/val/test splits
    db_documents_data = load_from_disk("data/raw/ORDERS_PARLIAMENT")

    # Concatenar train + validation + test
    db_documents_data["all"] = concatenate_datasets([
        db_documents_data["train"],
        db_documents_data["validation"],
        db_documents_data["test"]
    ])

    # save dataset and unique documents
    folder_path = os.path.join(FOLDER_PROCESSED, "parliament_all_docs")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    db_documents_data.save_to_disk(folder_path)
    print(f"Parliament dataset saved to {folder_path}")

    # instructions dataset
    qa_parliament = load_from_disk("data/raw/QA_PARLIAMENT_TRAIN")
    qa_parliament = qa_parliament.rename_columns({
        "PK": "id",
        "context": "documents",
        "answer": "response"
    })
    # quedarme solo con las columnas necesarias
    qa_parliament = qa_parliament.map(
        lambda x: {col: x[col] for col in columns_of_final_dataset}
    )
    qa_parliament_test = load_from_disk("data/raw/QA_PARLIAMENT_TEST")
    qa_parliament_test = qa_parliament_test.rename_columns({
        "PK": "id",
        "answer": "response"
    })
    def process_create_documents(example):
        # buscar el id en db_documents_data["all"] y obtener el context
        pk = example["id"]
        matched = db_documents_data["all"].filter(lambda x: x["PK"] == pk)
        if len(matched) > 0:
            example["documents"] = [matched[0]["text"]]
        else:
            raise ValueError(f"PK {pk} not found in db_documents_data")
        return example
    qa_parliament_test = qa_parliament_test.map(process_create_documents)
    # quedarme solo con las columnas necesarias
    qa_parliament_test = qa_parliament_test.map(
        lambda x: {col: x[col] for col in columns_of_final_dataset}
    )
    # concatenar train + test
    qa_parliament["test"] = qa_parliament_test["test"]
    folder_path = os.path.join(FOLDER_PROCESSED, "parliament_qa")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    qa_parliament.save_to_disk(folder_path)
    print(f"Parliament QA dataset saved to {folder_path}")


def download_squad():
    dataset_squad = load_dataset("rajpurkar/squad")

    titles = dataset_squad["validation"]["title"][:]
    context = dataset_squad["validation"]["context"][:]  

    dataset_knowledge = {
        "title": [],
        "text": []
    }
    for t, c in zip(titles, context):
        dataset_knowledge["title"].append(t)
        dataset_knowledge["text"].append(c)

    dataset_knowledge = Dataset.from_dict(dataset_knowledge)

    unique_texts = set()
    def is_unique(example):
        if example["text"] in unique_texts:
            return False
        unique_texts.add(example["text"])
        return True
    
    dataset_unique = dataset_knowledge.filter(is_unique)

    folder_path = os.path.join(FOLDER_PROCESSED, "squad_knowledge")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataset_unique.save_to_disk(folder_path)
    print(f"SQuAD knowledge dataset saved to {folder_path}")

    #split validation dataset into train, val, test (80%, 10%, 10%) create new dataset
    dataset_size = len(dataset_squad["validation"])
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    dataset_ = dataset_squad["validation"].shuffle(seed=SEED)
    train_dataset = dataset_.select(range(0, train_size))
    val_dataset = dataset_.select(range(train_size, train_size + val_size))
    test_dataset = dataset_.select(range(train_size + val_size, dataset_size))

    new_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })

    folder_path = os.path.join(FOLDER_PROCESSED, "squad_qa")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    new_dataset.save_to_disk(folder_path)
    print(f"SQuAD QA dataset saved to {folder_path}")

    # Dividir por dominios (titles) y mostrar número de ejemplos por dominio vamos a usar solo el European_Union_law
    domain = "European_Union_law"
    sud_domain_qa = dataset_squad['validation'].filter(lambda x: x['title'] == domain)
    train_test_split = sud_domain_qa.train_test_split(test_size=0.1) #test 10%, train 90%
    train_val_split = train_test_split['train'].train_test_split(test_size=0.1) #val 10%, train 90%
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    test_dataset = train_test_split['test']
    sub_dataset_domain = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    sud_domain_knowledge = dataset_unique.filter(lambda x: x['title'] == domain)
    # save sud_domain_qa and sud_domain_knowledge
    folder_path_qa = os.path.join(FOLDER_PROCESSED, f"squad_qa_{domain.lower()}")
    if not os.path.exists(folder_path_qa):
        os.makedirs(folder_path_qa)
    sub_dataset_domain.save_to_disk(folder_path_qa)
    print(f"SQuAD QA dataset for domain {domain} saved to {folder_path_qa}")
    folder_path_knowledge = os.path.join(FOLDER_PROCESSED, f"squad_knowledge_{domain.lower()}")
    if not os.path.exists(folder_path_knowledge):
        os.makedirs(folder_path_knowledge)
    sud_domain_knowledge.save_to_disk(folder_path_knowledge)
    print(f"SQuAD knowledge dataset for domain {domain} saved to {folder_path_knowledge}")

DATASETS_TO_DOWNLOAD = {
    "ragbench": download_ragbench,
    "clapnq": download_clapnq,
    "parliament": download_parliament,
    "squad": download_squad,
}


def main(args: argparse.Namespace):
    if args.dataset in DATASETS_TO_DOWNLOAD:
        DATASETS_TO_DOWNLOAD[args.dataset]()
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized. Available datasets: {list(DATASETS_TO_DOWNLOAD.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face using the datasets library.")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--split", type=str, default=None, help="Dataset split to download (e.g., 'train', 'test')")
    args = parser.parse_args()

    main(args)