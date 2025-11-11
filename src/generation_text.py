import os
import argparse
import json
import hashlib
from tqdm import tqdm
import re
from unsloth import FastLanguageModel
from vllm import SamplingParams
from datetime import datetime, timezone

from datasets import load_from_disk, Dataset

from evaluator import EvaluatorKnowledgeUsingAccuracy
from utils import setup_environment
from utils import (
    SEED as DEFAULT_SEED,
    SYSTEM_PROMPT_KNOWLEDGE,
    seed_everything,
)

# -----------------------------
# CLI Arguments
# -----------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch RAG generation with Unsloth + FAISS + vLLM")

    # Model / inference
    p.add_argument("--model", default=None,
                   help="Model name to load (HuggingFace/Unsloth). Use None to disable generation.")
    p.add_argument("--is_adapter", action="store_true", default=False,
                   help="Whether to use an adapter model.")
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="Maximum input sequence length.")
    p.add_argument("--max-generation-length", type=int, default=256,
                   help="Maximum number of tokens to generate.")
    p.add_argument("--max-lora-rank", type=int, default=512,
                   help="Maximum LoRA rank (only for adapters).")
    p.add_argument("--fast-inference", action="store_true", default=True,
                   help="Enable fast_inference in Unsloth (default ON).")
    p.add_argument("--load-in-4bit", action="store_true", default=False,
                   help="Load the model in 4-bit mode.")
    p.add_argument("--load-in-8bit", action="store_true", default=False,
                   help="Load the model in 8-bit mode.")

    # Data
    p.add_argument("--dataset", default="data/processed/squad_qa",
                   help="Path to the dataset (datasets.load_from_disk).")
    
    # Batch / sampling
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for generation.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature for vLLM.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, 
                   help="Random seed.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit the number of processed samples (optional).")

    # Output
    p.add_argument("--output-dir", default=None,
                   help="Base output folder. If not set, a structured path under results/ is used.")
    p.add_argument("--tag", default=None,
                   help="Optional tag to append to the run folder name (e.g., ablationA, grid1).")
    return p

# -----------------------------
# Utilities
# -----------------------------
def load_dataset(path: str) -> Dataset:
    ds = load_from_disk(path)["test"]
    print(f"Loaded {len(ds)} samples from {path}")
    return ds

def build_prompt(tokenizer, question: str) -> str:
    """Builds the chat prompt for a single example using the tokenizer chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_KNOWLEDGE},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\-\.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-_")

def short_alias(path_or_name: str) -> str:
    """Make a compact alias from a HF id or path (last segment, shortened)."""
    if not path_or_name:
        return "none"
    base = path_or_name.split("/")[-1]
    return slugify(base)

def make_expid(hparams_for_hash: dict) -> str:
    """Stable short hash from selected hyperparams."""
    keys = [
        "model", "load_in_4bit", "load_in_8bit", "lora_path",
        "max_seq_length", "max_generation_length", "temperature", "seed",
    ]
    payload = "|".join(f"{k}={hparams_for_hash.get(k)}" for k in keys)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

def now_utc_compact() -> str:
    # ISO-like but compact; Z = UTC
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = get_parser()
    args = parser.parse_args()

    is_adapter = args.is_adapter

    seed_everything(args.seed)
    setup_environment()

    evaluator = EvaluatorKnowledgeUsingAccuracy()

    # Aliases
    dataset_alias = slugify(os.path.basename(args.dataset))
    if is_adapter:
        lora_method = args.model.split("/")[-2]
        other_lora_settings = args.model.split("/")[-1]
        model_alias = slugify(f"{lora_method}_{other_lora_settings}")
    else:
        model_alias = short_alias(args.model)

    # Short signature for uniqueness
    expid = make_expid(vars(args))

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        load_in_8bit = args.load_in_8bit,
        fast_inference = args.fast_inference,
        max_lora_rank = args.max_lora_rank,
    )
    FastLanguageModel.for_inference(model)

    sampling_params = SamplingParams(
        temperature = args.temperature,
        max_tokens = args.max_generation_length,
        seed = args.seed,
    )

    timestamp = now_utc_compact()

    # Folder structure
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.join(
            "results",
            dataset_alias,
        )

    run_folder_name = f"{timestamp}_{model_alias}_{expid}"

    folder_output = os.path.join(base_dir, run_folder_name)
    os.makedirs(folder_output, exist_ok=True)

    print(f"[Run dir] {folder_output}")

    # Dataset
    dataset = load_from_disk(args.dataset)["validation"] # squad uses 'validation' split
    questions = dataset["question"]
    context = dataset["context"]
    golden_response = dataset["answers"]
    print(f"Using dataset from {args.dataset}")

    n_total = len(questions)
    n = min(args.limit, n_total) if args.limit is not None else n_total
    if n < n_total:
        print(f"Limiting to first {n} samples (of {n_total}).")

    # Save hparams.json (includes derived metadata)
    hparams = {
        "timestamp_utc": timestamp,
        "expid": expid,
        "dataset": args.dataset,
        "dataset_alias": dataset_alias,
        "model": args.model,
        "model_alias": model_alias,
        "is_adapter": is_adapter,
        "max_seq_length": args.max_seq_length,
        "max_generation_length": args.max_generation_length,
        "fast_inference": args.fast_inference,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "seed": args.seed,
        "limit": args.limit,
        "n_total_samples": n_total,
        "n_evaluated": n,
        "output_dir": folder_output,
    }
    with open(os.path.join(folder_output, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    records = []

    # Batch loop
    for start in tqdm(range(0, n, args.batch_size), desc="Processing batches"):
        end = min(start + args.batch_size, n)
        batch_user_input = questions[start:end]
        batch_contexts = context[start:end]
        batch_reference = golden_response[start:end]

        # Prompts
        batch_prompts = [
            build_prompt(tokenizer, q)
            for q in batch_user_input
        ]
        # Batch generation
        batch_outputs = model.fast_generate(
            batch_prompts,
            sampling_params = sampling_params,
            lora_request = model.load_lora(args.model) if is_adapter else None,
        )
        # Extract text (keep order)
        batch_texts = [out.outputs[0].text for out in batch_outputs]

        # Collect records
        for i in range(len(batch_user_input)):
            records.append({
                "user_input": batch_user_input[i],
                "context": batch_contexts[i],
                "response": batch_texts[i],      # None in retrieval-only
                "reference": batch_reference[i]["text"][0],  # first reference answer
            })

    # Evaluation (Generation)
    print("Running generation evaluation...")
    results, df_results = evaluator.evaluate(records)
    print(results)
    # Save results
    with open(os.path.join(folder_output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    df_results.to_csv(os.path.join(folder_output, "generation_results.csv"), index=False)
    print("Saved results.")

if __name__ == "__main__":
    main()

