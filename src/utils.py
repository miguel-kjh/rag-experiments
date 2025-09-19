import os
import random
import numpy as np
import torch

FOLDER_DATA      = "data"
FOLDER_RAW       = os.path.join(FOLDER_DATA, "raw")
FOLDER_PROCESSED = os.path.join(FOLDER_DATA, "processed")
FOLDER_COMBINED  = os.path.join(FOLDER_DATA, "combined")
FOLDER_DB        = os.path.join(FOLDER_DATA, "db")

RANGBENCH_SUBSETS = [
    'covidqa', 'cuad', 'delucionqa', 'emanual', 'expertqa', 'finqa', 'hagrid', 'hotpotqa', 'msmarco', 'pubmedqa', 'tatqa', 'techqa'
]

SEED             = 123

SYSTEM_PROMPT     = "You are a helpful AI assistant. Use the provided context to answer questions."
PROMPT            = """Answer the question based only on the following context:
Context: {context}

Question: {question}

Answer:"""

def seed_everything(seed):
    """Set seed for reproducibility."""
    print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # create folders if they don't exist
    for folder in [FOLDER_DATA, FOLDER_RAW, FOLDER_PROCESSED, FOLDER_COMBINED, FOLDER_DB]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

if __name__ == "__main__":
    main()

