from datasets import load_from_disk

FOLDER_AUTORE = "data/processed/parliament_all_docs"
dataset_indexing = load_from_disk(FOLDER_AUTORE)["all"]
dataset_indexing = dataset_indexing.rename_column("PK", "label")

num_labels = len(dataset_indexing['label'])
print(f"Number of labels: {num_labels}")
labels_list = dataset_indexing.unique('label')
print(f"Labels: {labels_list}")

# map labels to integers
label_to_id = {label: i for i, label in enumerate(labels_list)}
def map_labels(example):
    return {
        "label": label_to_id[example['label']]
    }
dataset_indexing = dataset_indexing.map(map_labels)
print(dataset_indexing[0])


dataset_query = load_from_disk("data/processed/parliament_qa")

print(dataset_query["train"][0])