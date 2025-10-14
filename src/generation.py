import os
import argparse
import json
import hashlib
from tqdm import tqdm
import re
from unsloth import FastLanguageModel
from vllm import SamplingParams
from rafa import Rafa
from datetime import datetime, timezone
from typing import Tuple, List, Optional

from datasets import load_from_disk, Dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ranking_metrics import calc_ranking_metrics
from query_expansion import QueryDescomposition, QueryRewriter, HyDEGenerator
from retriever import Retriever, NaiveDenseRetriever, HybridRetriever
from reranker import Reranker, CrossEncoderReranker
from embeddings_models import SentenceTransformerEmbeddings
from utils import setup_environment
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

def retrieve_documents(query: str, retriever: Retriever, reranker: Reranker) -> Tuple[str, List[str], List[str]]:
    """Retrieve top-k relevant documents from the vector index and concatenate contents."""
    results = retriever.retrieve(query)

    if reranker:
        results = reranker.rerank(query, results)
        if type(reranker) is Rafa:
            # Rafa returns (doc, score, reasoning)
            results = [doc for doc, score, reasoning in results]
        else:
            # Other rerankers return (doc, score)
            results = [doc for doc, score in results]
    
    list_contents = [doc.page_content for doc in results]
    context = "\n".join(list_contents)
    idx = [doc.metadata.get("id", doc.metadata.get("doc_id", None)) for doc in results]
    return context, idx, list_contents

def retrival_documents_query_expansion(
        real_query: str, 
        query_transformed: Optional[str|List[str]], 
        retriever: Retriever, 
        reranker: Reranker
    ) -> Tuple[str, List[str], List[str]]:
    """Retrieve top-k relevant documents from the vector index and concatenate contents."""
    if isinstance(query_transformed, str):
        results = retriever.retrieve(query_transformed)
    elif isinstance(query_transformed, list):
        results = retriever.retrieve(real_query)  # always include original query
        for q in query_transformed:
            res = retriever.retrieve(q)
            results.extend(res)
        # quitar duplicados pero mantener orden
        seen = set()
        unique_results = []
        for doc in results:
            doc_id = doc.metadata.get("id", doc.metadata.get("doc_id", None))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        results = unique_results
    else:
        raise ValueError("Invalid query_transformed type")
        
    if reranker:
        results = reranker.rerank(real_query, results)
        results = [doc for doc, score in results]
    
    list_contents = [doc.page_content for doc in results]
    context = "\n".join(list_contents)
    idx = [doc.metadata.get("id", doc.metadata.get("doc_id", None)) for doc in results]
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
    if model_value is None:
        return True
    mv = str(model_value).strip().lower()
    return mv in {"none", "null", ""}

def using_pre_retrieval(expansion_method: str) -> bool:
    em = expansion_method.strip().lower()
    return em in {"rewriter", "hyde", "multiquery"}

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
    base = base.replace("sentence-transformers-", "").replace("paraphrase-", "")
    return slugify(base)

def make_expid(hparams_for_hash: dict) -> str:
    """Stable short hash from selected hyperparams."""
    keys = [
        "db_path", "embedding_model", "top_k", "model", "temperature",
        "max_seq_length", "max_generation_length", "seed"
    ]
    payload = "|".join(f"{k}={hparams_for_hash.get(k)}" for k in keys)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

def now_utc_compact() -> str:
    # ISO-like but compact; Z = UTC
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

# -----------------------------
# CLI Arguments
# -----------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch RAG generation with Unsloth + FAISS + vLLM")

    # Model / inference
    p.add_argument("--model", default=None,
                   help="Model name to load (HuggingFace/Unsloth). Use None to disable generation.")
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="Maximum input sequence length.")
    p.add_argument("--max-generation-length", type=int, default=1024,
                   help="Maximum number of tokens to generate.")
    p.add_argument("--fast-inference", action="store_true", default=True,
                   help="Enable fast_inference in Unsloth (default ON).")
    p.add_argument("--load-in-4bit", action="store_true", default=False,
                   help="Load the model in 4-bit mode.")
    p.add_argument("--load-in-8bit", action="store_true", default=False,
                   help="Load the model in 8-bit mode.")

    # Data
    p.add_argument("--dataset", default="data/processed/parliament_qa",
                   help="Path to the dataset (datasets.load_from_disk).")
    p.add_argument("--db-path", default="data/db/parliament_db/parliament_all_docs_embeddings_sentence-transformers_paraphrase-multilingual-mpnet-base-v2",
                   help="Path to the FAISS index.")
    
    # Pre-retrieval
    p.add_argument("--expansion-method", default="none",
                   help="Query expansion method: 'none', 'rewriter', or 'hyde'.")
    p.add_argument("--expansion-model", default="Qwen/Qwen3-0.6B",
                     help="Model name for query expansion (if enabled).")
    p.add_argument("--expansion-enable-thinking", action="store_true", default=False,
                   help="Enable chain-of-thought (thinking) in query expansion (default False).")

    # Retriever type
    p.add_argument("--retriever-type", default="dense",
                   help="Type of retriever to use: 'dense' or 'hybrid'.")
    p.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                   help="Embedding model used for FAISS.")
    p.add_argument("--top-k", type=int, default=4,
                   help="Number of documents to retrieve per query.")
    p.add_argument("--similarity_function", default="similarity",
                   help="Similarity function to use (similarity or mmr).")
    p.add_argument("--lambda-mult", type=float, default=None,
                   help="Lambda multiplier for MMR (if using mmr).")
    p.add_argument("--sparse-retriever", default="bm25",
                   help="If retriever-type is 'hybrid', choose sparse retriever: 'bm25' or 'tfidf'.")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="If retriever-type is 'hybrid', alpha weight for dense retriever (0 to 1).")
    
    # Reranker (not used in main script, but could be integrated)
    p.add_argument("--reranker-model", default=None,
                   help="Cross-encoder model for reranking (optional). E.g., 'BAAI/bge-reranker-v2-m3'.")
    p.add_argument("--top-rank", type=int, default=5,
                   help="Number of top documents to keep after reranking.")
    p.add_argument("--use-chunking", action="store_true", default=False,
                   help="Whether to use chunking in the reranker (default False).")

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
                   help="Base output folder. If not set, a structured path under results/ is used.")
    p.add_argument("--tag", default=None,
                   help="Optional tag to append to the run folder name (e.g., ablationA, grid1).")
    return p

# -----------------------------
# Main
# -----------------------------
def main():
    parser = get_parser()
    args = parser.parse_args()

    seed_everything(args.seed)
    setup_environment()

    retrieval_only = is_no_model(args.model)
    using_expansion = using_pre_retrieval(args.expansion_method)
    task = "retrieval" if retrieval_only else "rag"

    # Aliases
    dataset_alias = slugify(os.path.basename(args.dataset))
    db_alias = slugify(os.path.basename(args.db_path))
    emb_alias = short_alias(args.embedding_model)
    model_alias = short_alias(args.model) if not retrieval_only else "none"

    # Short signature for uniqueness
    expid = make_expid({
        "db_path": os.path.basename(args.db_path),
        "embedding_model": args.embedding_model,
        "top_k": args.top_k,
        "similarity_function": args.similarity_function,
        "lambda_mult": args.lambda_mult,
        "sparse_retriever": args.sparse_retriever,
        "alpha": args.alpha,
        "reranker_model": args.reranker_model,
        "top_rank": args.top_rank,
        "use_chunking": args.use_chunking,
        "model": args.model,
        "temperature": args.temperature,
        "max_seq_length": args.max_seq_length,
        "max_generation_length": args.max_generation_length,
        "seed": args.seed,
    })

    # Model (optional)
    model = None
    tokenizer = None
    sampling_params = None

    if retrieval_only:
        print("Running in RETRIEVAL-ONLY mode: no LLM will be loaded and no text will be generated.")

    if using_expansion:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = args.expansion_model,
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

    print("### RAG COMPONENTS ###")

    # Vector DB for RAG
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": "cuda"},
    )
    db = FAISS.load_local(
        args.db_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded FAISS index from {args.db_path}")

    # Query Expansion
    query_expander = None
    if using_expansion:
        lang = "es" if "parliament" in dataset_alias else "en"
        if args.expansion_method.strip().lower() == "rewriter":
            query_expander = QueryRewriter(
                llm_model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                enable_thinking=args.expansion_enable_thinking,
                lang=lang
            )
        elif args.expansion_method.strip().lower() == "hyde":
            query_expander = HyDEGenerator(
                llm_model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                enable_thinking=args.expansion_enable_thinking,
                lang=lang
            )
        elif args.expansion_method.strip().lower() == "multiquery":
            query_expander = QueryDescomposition(
                llm_model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                enable_thinking=args.expansion_enable_thinking,
                lang=lang
            )
        else:
            raise ValueError("expansion-method must be 'multiquery', 'rewriter', or 'hyde'")
        print(f"Using query expansion method: {query_expander}")

    # Retriever
    if args.retriever_type == "dense":
        retriever = NaiveDenseRetriever(
            db=db,
            top_k=args.top_k,
            search_type=args.similarity_function,
            lambda_mult=args.lambda_mult,
        )
    elif args.retriever_type == "hybrid":
        retriever = HybridRetriever(
            db=db,
            sparse_retriever_name=args.sparse_retriever,
            top_k=args.top_k,
            alpha=args.alpha,
            search_type=args.similarity_function,
            lambda_mult=args.lambda_mult,
        )
    else:
        raise ValueError("retriever-type must be 'dense' or 'hybrid'")
    print(f"Using retriever: {retriever}")

    reranker = None
    if args.reranker_model:
        if args.reranker_model.lower() == "rafa":
            reranker = Rafa(
                model_name="Qwen/Qwen3-0.6B",
                max_seq_length=8192,
                max_new_tokens=1024,
                use_chunking=False,
                batch_size=args.batch_size,
                load_in_4bit=args.load_in_4bit,
                top_rank=args.top_rank
            )
        else:
            reranker = CrossEncoderReranker(
                model_name=args.reranker_model,
                top_rank=args.top_rank,
                use_chunking=args.use_chunking
            )
        print(f"Reranker enabled: {reranker}")

    print("\n### RAG COMPONENTS ###")

    timestamp = now_utc_compact()
    retriever_sig = f"R@k{args.top_k}-{emb_alias}"

    # Folder structure
    if args.output_dir:
        base_dir = args.output_dir
    else:
        base_dir = os.path.join(
            "results",
            dataset_alias,
            db_alias,
            retriever_sig,
        )

    run_folder_name = f"{timestamp}_{task}_{query_expander}_{retriever}_{reranker}_{model_alias}_{expid}"
    if args.tag:
        run_folder_name += f"_{slugify(args.tag)}"

    folder_output = os.path.join(base_dir, run_folder_name)
    os.makedirs(folder_output, exist_ok=True)

    print(f"[Run dir] {folder_output}")

    # Dataset
    dataset = load_from_disk(args.dataset)["test"]
    questions = dataset["question"]
    golden_response = dataset["response"]
    try:
        golden_document_ids = dataset["document_ids"]
    except ValueError:
        golden_document_ids = dataset["id"]
        golden_document_ids = [[x] for x in golden_document_ids]
    print(f"Using dataset from {args.dataset}")

    n_total = len(questions)
    n = min(args.limit, n_total) if args.limit is not None else n_total
    if n < n_total:
        print(f"Limiting to first {n} samples (of {n_total}).")

    # Save hparams.json (includes derived metadata)
    hparams = {
        "timestamp_utc": timestamp,
        "task": task,
        "expid": expid,
        "tag": args.tag,
        "dataset": args.dataset,
        "dataset_alias": dataset_alias,
        "db_path": args.db_path,
        "db_alias": db_alias,
        "expansion_method": args.expansion_method,
        "expansion_model": args.expansion_model,
        "expansion_enable_thinking": args.expansion_enable_thinking,
        "using_expansion": using_expansion,
        "embedding_model": args.embedding_model,
        "embedding_alias": emb_alias,
        "top_k": args.top_k,
        "similarity_function": args.similarity_function,
        "lambda_mult": args.lambda_mult,
        "sparse_retriever": args.sparse_retriever,
        "alpha": args.alpha,
        "reranker_model": args.reranker_model,
        "top_rank": args.top_rank,
        "use_chunking": args.use_chunking,
        "model": args.model,
        "model_alias": model_alias,
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
        "retriever_signature": retriever_sig,
        "output_dir": folder_output,
    }
    with open(os.path.join(folder_output, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    records = {}

    # Batch loop
    for start in tqdm(range(0, n, args.batch_size), desc="Processing batches"):
        end = min(start + args.batch_size, n)
        batch_user_input = questions[start:end]
        batch_reference = golden_response[start:end]
        batch_target_document_ids = golden_document_ids[start:end]

        # Batch retrieval
        
        if using_expansion:
            batch_user_input_expanded = query_expander.expand(batch_user_input)
            batch_contexts = [retrival_documents_query_expansion(q, qe, retriever, reranker) for q, qe in zip(batch_user_input, batch_user_input_expanded)]
        else:
            batch_contexts = [retrieve_documents(q, retriever, reranker) for q in batch_user_input]

        if retrieval_only:
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
        #print(f"Processed {end}/{n} samples.")

    # Save results
    filename = "retrieval.jsonl" if retrieval_only else "generation.jsonl"
    output_path = os.path.join(folder_output, filename)
    print(f"Saving results to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for key in sorted(records.keys()):
            f.write(json.dumps(records[key], ensure_ascii=False) + "\n")

    # Evaluation (ReID)
    print("Running ranking evaluation...")
    preds = [records[k]["document_ids"] for k in sorted(records.keys())]
    refs = [records[k]["target_document_ids"] for k in sorted(records.keys())]
    if "parliament" in args.dataset.lower():
        # Parliament QA has 1 relevant document per query
        metrics = calc_ranking_metrics(preds, refs, one_relevant_per_query=True, include_classification_view=True)
    else:
        # RAG-Bench datasets have multiple relevant documents per query
        metrics = calc_ranking_metrics(preds, refs, one_relevant_per_query=False, include_classification_view=True)
    print("Ranking results:")
    print(metrics)
    with open(os.path.join(folder_output, "ranking_results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

