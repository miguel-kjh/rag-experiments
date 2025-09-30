import os
import argparse
import json
from typing import Tuple, List, Optional

from datasets import load_from_disk, Dataset
from langchain_community.vectorstores import FAISS

from embeddings_models import SentenceTransformerEmbeddings
from utils import (
    SEED as DEFAULT_SEED,
    SYSTEM_PROMPT,
    PROMPT,
    seed_everything,
)

# -----------------------------
# Utilities
# -----------------------------
def load_dataset(path: str) -> Dataset:
    ds = load_from_disk(path)["test"]
    print(f"Loaded {len(ds)} samples from {path}")
    return ds

def retrieve(query: str, db: FAISS, k: int) -> Tuple[str, List[str], List[str]]:
    """Retrieve top-k relevant documents from the vector index and concatenate contents."""
    results = db.similarity_search(query, k=k)  # L2 similarity
    list_contents = [doc.page_content for doc in results]
    context = "\n".join(list_contents)
    idx = [doc.metadata["id"] for doc in results]
    return context, idx, list_contents

def build_prompt(tokenizer, question: str, context: str) -> str:
    """Builds the chat prompt for a single example using the tokenizer chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT.format(context=context, question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

def is_no_model(model_value: Optional[str]) -> bool:
    """Return True if the CLI intends to disable the LLM."""
    if model_value is None:
        return True
    mv = str(model_value).strip().lower()
    return mv in {"none", "null", ""}

# -----------------------------
# CLI Arguments
# -----------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch RAG generation with Unsloth + FAISS + vLLM")

    # Model / inference
    p.add_argument("--model", default=None,
                   help="Model name to load (HuggingFace/Unsloth). Use None to disable generation.")
    p.add_argument("--max-seq-length", type=int, default=32000,
                   help="Maximum input sequence length.")
    p.add_argument("--max-generation-length", type=int, default=1024,
                   help="Maximum number of tokens to generate.")
    p.add_argument("--fast-inference", action="store_true", default=True,
                   help="Enable fast_inference in Unsloth (default ON).")
    p.add_argument("--load-in-4bit", action="store_true", default=False,
                   help="Load the model in 4-bit mode.")
    p.add_argument("--load-in-8bit", action="store_true", default=False,
                   help="Load the model in 8-bit mode.")

    # Data / RAG
    p.add_argument("--dataset", default="data/processed/parliament_qa",
                   help="Path to the dataset (datasets.load_from_disk).")
    p.add_argument("--db-path", default="data/db/parliament_db/parliament_all_docs_embeddings_sentence-transformers_paraphrase-multilingual-mpnet-base-v2",
                   help="Path to the FAISS index.")
    p.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                   help="Embedding model used for FAISS.")
    p.add_argument("--top-k", type=int, default=4,
                   help="Number of documents to retrieve per query.")

    # Batch / sampling
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for generation.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for vLLM.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, 
                   help="Random seed.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit the number of processed samples (optional).")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Output folder. Default: results/generation_{dbbasename}_{modelname}.")
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)

    # Decide mode
    retrieval_only = is_no_model(args.model)

    # Experiment name / output folder
    db_base = os.path.basename(args.db_path)
    model_suffix = (args.model.split('/')[-1] if not retrieval_only else "retrieval_only")
    filename_of_experiment = f"{db_base}_{model_suffix}"
    folder_output = args.output_dir or f"results/generation_{filename_of_experiment}"
    os.makedirs(folder_output, exist_ok=True)

    # Model (optional)
    model = None
    tokenizer = None
    sampling_params = None

    if retrieval_only:
        print("Running in RETRIEVAL-ONLY mode: no LLM will be loaded and no text will be generated.")
    else:
        from unsloth import FastLanguageModel
        from vllm import SamplingParams 
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.model,
            max_seq_length = args.max_seq_length,
            load_in_4bit = args.load_in_4bit,
            load_in_8bit = args.load_in_8bit,
            fast_inference = args.fast_inference,
        )
        sampling_params = SamplingParams(
            temperature = args.temperature,
            max_tokens = args.max_generation_length,
            seed = args.seed,
        )

    # Vector DB for RAG
    embedding_model = SentenceTransformerEmbeddings(args.embedding_model, device='cuda')
    db = FAISS.load_local(
        args.db_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded FAISS index from {args.db_path}")

    # Dataset
    dataset = load_from_disk(args.dataset)["test"]
    questions = dataset["question"]
    golden_response = dataset["response"]
    try:
        golden_document_ids = dataset["document_ids"]
    except ValueError:
        golden_document_ids = dataset["id"]
        golden_document_ids = [[x] for x in golden_document_ids]

    n_total = len(questions)
    n = min(args.limit, n_total) if args.limit is not None else n_total
    if n < n_total:
        print(f"Limiting to first {n} samples (of {n_total}).")

    records = {}

    # Batch loop
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_user_input = questions[start:end]
        batch_reference = golden_response[start:end]
        batch_target_document_ids = golden_document_ids[start:end]

        # Batch retrieval
        batch_contexts = [retrieve(q, db, args.top_k) for q in batch_user_input]

        if retrieval_only:
            # No prompt building, no generation
            batch_texts = [None] * len(batch_user_input)  # explicit null in JSON
        else:
            # Prompts
            batch_prompts = [
                build_prompt(tokenizer, q, c)
                for q, (c, _, _) in zip(batch_user_input, batch_contexts)
            ]

            # Batch generation
            batch_outputs = model.fast_generate(
                batch_prompts,
                sampling_params = sampling_params,
                lora_request = None,
            )

            # Extract text (keep order)
            batch_texts = [out.outputs[0].text for out in batch_outputs]

        # Collect records
        for i in range(len(batch_user_input)):
            records[start + i] = {
                "user_input": batch_user_input[i],
                "document_ids": batch_contexts[i][1],
                "target_document_ids": batch_target_document_ids[i],
                "retrieved_contexts": batch_contexts[i][2],
                "response": batch_texts[i],      # None in retrieval-only
                "reference": batch_reference[i],
            }

        print(f"Processed {end}/{n} samples.")

    # Save results
    generation_filename = os.path.join(folder_output, "generation.jsonl")
    print(f"Saving generation results to {generation_filename}")
    with open(generation_filename, "w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            f.write(json.dumps(records[key], ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
