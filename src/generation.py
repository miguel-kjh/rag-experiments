from unsloth import FastLanguageModel
# from transformers import TextStreamer  # <- No se usa en batch, puedes quitarlo
from datasets import load_from_disk, Dataset
from langchain_community.vectorstores import FAISS
from vllm import SamplingParams

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
DATASET = "data/processed/legal-qa-v1"
DB_PATH = "data/db/legal-qa-v1_embeddings_all-mpnet-base-v2"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K = 1      # nº de documentos a recuperar
BATCH_SIZE = 32  # <<--- batch size de generación
FOLDER_OUTPUT = "results/generation.jsonl"

# -----------------------------
# Utilidades
# -----------------------------
def load_dataset(path: str) -> Dataset:
    ds = load_from_disk(path)["test"]
    print(f"Loaded {len(ds)} samples from {path}")
    return ds

def retrieve(query: str, db: FAISS, k: int = TOP_K) -> str:
    """Recupera documentos relevantes del índice vectorial y concatena sus contenidos."""
    results = db.similarity_search(query, k=k)  # L2/Cosine según el índice
    return "\n".join([doc.page_content for doc in results])

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
    answers = dataset["answer"]
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
        batch_questions = questions[start:end]
        batch_answers = answers[start:end]

        # Recuperación por lote
        batch_contexts = [retrieve(q, db) for q in batch_questions]

        # Prompts por muestra
        batch_prompts = [
            build_prompt(tokenizer, q, c)
            for q, c in zip(batch_questions, batch_contexts)
        ]

        # Generación en batch (lista de prompts)
        batch_outputs = model.fast_generate(
            batch_prompts,
            sampling_params = sampling_params,
            lora_request = None,
        )

        # Extraer texto (manteniendo el orden del batch)
        batch_texts = [out.outputs[0].text for out in batch_outputs]

        for i in range(len(batch_questions)):
            q = batch_questions[i]
            a = batch_answers[i]
            g = batch_texts[i]
            records[start + i] = {
                "question": q,
                "answer": a,
                "generation": g,
                "context": batch_contexts[i],
            }

        print(f"Processed {end}/{n} samples.")

    print(records[0])

    # Guardar resultados
    import json
    with open(FOLDER_OUTPUT, "w") as f:
        for record in records.values():
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
