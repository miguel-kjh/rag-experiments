import os
import argparse
import pandas as pd
from typing import Dict, Optional, Any

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    AnswerAccuracy,
    ContextRelevance,
    ResponseGroundedness,
    Faithfulness,
    FactualCorrectness,
    BleuScore,
    RougeScore,
)
from langchain_openai import ChatOpenAI

from src.ranking_metrics import calc_ranking_metrics
from utils import SEED, setup_environment


# -----------------------------
# Components
# -----------------------------
def build_components(model_name: str):
    """Initialize evaluation components."""
    evaluator_llm = ChatOpenAI(model=model_name, seed=SEED)
    return evaluator_llm


# -----------------------------
# Data loading
# -----------------------------
def load_dataset(path: str, subsample: Optional[int] = None):
    """Load dataset from JSONL and optionally subsample it."""
    dataset = pd.read_json(path, lines=True).to_dict(orient="records")
    if subsample:
        dataset = dataset[:subsample]
    print(f"Loaded dataset with {len(dataset)} samples.")
    return dataset


# -----------------------------
# Evaluations
# -----------------------------
def run_llm_evaluation(dataset, evaluator_llm):
    """Run evaluation on the dataset using predefined metrics."""
    ragas_dataset = EvaluationDataset.from_list(dataset)
    metrics = [
        AnswerAccuracy(),
        ContextRelevance(),
        ResponseGroundedness(),
        Faithfulness(),
        FactualCorrectness(),
        BleuScore(),
        RougeScore(rouge_type="rougeL"),
    ]
    return evaluate(dataset=ragas_dataset, metrics=metrics, llm=evaluator_llm)


def run_ranking_evaluation(dataset, one_relevant_per_query: bool = True) -> Dict:
    """Run re-identification metrics evaluation."""
    preds = [item["document_ids"] for item in dataset]
    gts = [item["target_document_ids"] for item in dataset]
    return calc_ranking_metrics(preds, gts, one_relevant_per_query=one_relevant_per_query)


# -----------------------------
# Saving results
# -----------------------------
def save_results(final_results: Dict, llm_results: Optional[Any], output_path: str, output_record_path: str):
    """Save evaluation results (summary JSONL + detailed CSV if available)."""
    if llm_results is not None:
        df = llm_results.to_pandas()
        df.to_csv(output_record_path, index=False)
        print(f"Saved detailed evaluation records to {output_record_path}")
    # Save summary results
    with open(output_path, "w") as f:
        f.write(str(final_results))
    print(f"Saved summary evaluation results to {output_path}")


def join_results(reid_results: Dict, llm_results) -> Dict:
    """Join re-identification results with LLM evaluation results."""
    score_strs = {k: round(v, 4) for k, v in llm_results._repr_dict.items()}
    final_result = score_strs.copy()
    for k, v in reid_results.items():
        final_result[k] = round(v, 4)
    return final_result


# -----------------------------
# CLI parser
# -----------------------------
def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RAG generations with ReID metrics and LLM-based metrics.")
    p.add_argument("--dataset-path", default="results/generation_parliament_all_docs_embeddings_sentence-transformers_paraphrase-multilingual-mpnet-base-v2_retrieval_only",
                   help="Path to the dataset folder containing generation.jsonl.")
    p.add_argument("--generation-file", default="generation.jsonl",
                   help="Name of the generation JSONL file (inside dataset folder).")
    p.add_argument("--output-file", default="evaluation_results.jsonl",
                   help="Filename for summary results.")
    p.add_argument("--output-record-file", default="detailed_evaluation_records.csv",
                   help="Filename for detailed LLM evaluation records.")
    p.add_argument("--model-name", default="gpt-4o-mini",
                   help="LLM model used for evaluation (via OpenAI).")
    p.add_argument("--subsample-size", type=int, default=None,
                   help="Limit the dataset to the first N samples (for quick testing).")
    p.add_argument("--run-llm-evaluation", action="store_true",
                   help="Also run LLM-based evaluation metrics (Ragas).")
    return p


# -----------------------------
# Main
# -----------------------------
def main():
    parser = get_parser()
    args = parser.parse_args()

    setup_environment()

    dataset_path = args.dataset_path
    generation_path = os.path.join(dataset_path, args.generation_file)
    output_path = os.path.join(dataset_path, args.output_file)
    output_record_path = os.path.join(dataset_path, args.output_record_file)

    # Load dataset
    dataset = load_dataset(generation_path, args.subsample_size)

    # Run ranking evaluation
    # TODO: Posiblemente borrar la evaluacion del ranking ya lo hago cuando se generan las respuestas
    if "parliament" in args.dataset_path.lower():
        # Parliament QA has 1 relevant document per query
        one_relevant_per_query = True
    else:
        # RAG-Bench datasets have multiple relevant documents per query
        one_relevant_per_query = False
    final_result = run_ranking_evaluation(dataset, one_relevant_per_query=one_relevant_per_query)

    # Optionally run LLM-based evaluation
    if args.run_llm_evaluation:
        evaluator_llm = build_components(args.model_name)
        llm_result = run_llm_evaluation(dataset, evaluator_llm)
        final_result = join_results(final_result, llm_result)
        print("Final Evaluation Results:", final_result)
        save_results(final_result, llm_result, output_path, output_record_path)
    else:
        print("ReID Evaluation Results:", final_result)
        save_results(final_result, None, output_path, output_record_path)


if __name__ == "__main__":
    main()
