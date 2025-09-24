import os
from unsloth import FastLanguageModel
# from transformers import TextStreamer  # <- No se usa en batch, puedes quitarlo
from datasets import load_from_disk, Dataset
from langchain_community.vectorstores import FAISS
from vllm import SamplingParams
from typing import Tuple, List
import json

from embeddings_models import SentenceTransformerEmbeddings
from utils import (
    SEED,
    SYSTEM_PROMPT,
    PROMPT,
    seed_everything,
)

# -----------------------------
# Config
# -----------------------------
MODEL_GENERATION = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 8192
MAX_GENERATION_LENGTH = 512
DATASET = "data/raw/ragbench-covidqa"
DB_PATH = "data/db/ragbench-covidqa/ragbench-covidqa_embeddings_all-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 4      # nº de documentos a recuperar
BATCH_SIZE = 32  # <<--- batch size de generación
filename_of_experiment = os.path.basename(DB_PATH) + "_" + MODEL_GENERATION.split("/")[-1]
FOLDER_OUTPUT = f"results/generation_{filename_of_experiment}"

# -----------------------------
# Utilidades
# -----------------------------
def load_dataset(path: str) -> Dataset:
    ds = load_from_disk(path)["test"]
    print(f"Loaded {len(ds)} samples from {path}")
    return ds

def retrieve(query: str, db: FAISS, k: int = TOP_K) -> Tuple[str, List[str]]:
    """Recupera documentos relevantes del índice vectorial y concatena sus contenidos."""
    results = db.similarity_search(query, k=k)  # L2 similarity
    list_contents = [doc.page_content for doc in results]
    context = "\n".join([doc.page_content for doc in results])
    idx = [doc.metadata["id"] for doc in results]
    return context, idx, list_contents

def build_prompt(tokenizer, question: str, context: str) -> str:
    """Construye el prompt chat por muestra aplicando la chat template del tokenizer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT.format(context=context, question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

# -----------------------------
# Main
# -----------------------------
def main():
    seed_everything(SEED)

    # Modelo
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_GENERATION,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = True,
    )

    # Vector DB para RAG
    embedding_model = SentenceTransformerEmbeddings(EMBEDDING_MODEL, device='cuda')
    db = FAISS.load_local(
        DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    # Dataset
    dataset = load_dataset(DATASET)
    questions = dataset["question"]
    golden_response = dataset["response"]
    golden_document_ids = dataset["document_ids"]
    n = len(questions)

    # Parámetros de muestreo (vLLM)
    sampling_params = SamplingParams(
        temperature = 0.0,
        max_tokens = MAX_GENERATION_LENGTH,
        seed = SEED,
    )

    records = {}

    # Generación por lotes
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_user_input = questions[start:end]
        batch_reference = golden_response[start:end]
        batch_target_document_ids = golden_document_ids[start:end]

        # Recuperación por lote
        batch_contexts = [retrieve(q, db) for q in batch_user_input]

        # Prompts por muestra
        batch_prompts = [
            build_prompt(tokenizer, q, c)
            for q, (c, _, _) in zip(batch_user_input, batch_contexts)
        ]

        # Generación en batch (lista de prompts)
        batch_outputs = model.fast_generate(
            batch_prompts,
            sampling_params = sampling_params,
            lora_request = None,
        )

        # Extraer texto (manteniendo el orden del batch)
        batch_texts = [out.outputs[0].text for out in batch_outputs]

        for i in range(len(batch_user_input)):
            records[start + i] = {
                "user_input": batch_user_input[i],
                "document_ids": batch_contexts[i][1],
                "target_document_ids": batch_target_document_ids[i],
                "retrieved_contexts": batch_contexts[i][2],
                "response": batch_texts[i],
                "reference": batch_reference[i],
            }

        print(f"Processed {end}/{n} samples.")

    # Guardar resultados
    # crear folder_output si no existe
    os.makedirs(FOLDER_OUTPUT, exist_ok=True)
    generation_filename = os.path.join(FOLDER_OUTPUT, "generation.jsonl")
    print(f"Saving generation results to {generation_filename}")
    with open(generation_filename, "w") as f:
        for record in records.values():
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
