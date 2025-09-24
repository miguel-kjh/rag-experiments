import json
import os
from typing import Dict
import openai
import pandas as pd

from dotenv import load_dotenv
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
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from reid_metrics import calc_reid_metrics
from utils import SEED

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "results/generation_ragbench-covidqa_embeddings_all-mpnet-base-v2_Llama-3.2-1B-Instruct"
GENERATION_PATH = os.path.join(DATASET_PATH, "generation.jsonl")
OUTPUT_PATH = os.path.join(DATASET_PATH, "evaluation_results.jsonl")
OUTPUT_RECORD_PATH = os.path.join(DATASET_PATH, "detailed_evaluation_records.csv")
MODEL_NAME = "gpt-4o-mini"
SUBSAMPLE_SIZE = None  # Set to an integer for quick testing, or None to use full dataset


def setup_environment():
    """Load environment variables and set the OpenAI API key."""
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


def build_components():
    """Initialize evaluation components"""
    evaluator_llm = ChatOpenAI(model=MODEL_NAME, seed=SEED)
    return evaluator_llm


def load_dataset(path: str, subsample: int = None):
    """Load dataset from JSONL and optionally subsample it."""
    #dataset = EvaluationDataset.from_jsonl(path)
    dataset = pd.read_json(path, lines=True).to_dict(orient="records")
    if subsample:
        dataset = dataset[:subsample]
    print(f"Loaded dataset with {len(dataset)} samples.")
    return dataset


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

def run_reid_evaluation(dataset):
    """Run re-identification metrics evaluation."""
    preds = [item["document_ids"] for item in dataset]
    gts = [item["target_document_ids"] for item in dataset]
    return calc_reid_metrics(preds, gts)


def save_results(final_results: Dict, llm_results, output_path: str, output_record_path: str):
    """Convert evaluation results to CSV and save them."""
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

def main():
    setup_environment()
    evaluator_llm = build_components()
    dataset = load_dataset(GENERATION_PATH, SUBSAMPLE_SIZE)
    reid_result = run_reid_evaluation(dataset)
    llm_result = run_llm_evaluation(dataset, evaluator_llm)
    final_result = join_results(reid_result, llm_result)
    print("Final Evaluation Results:", final_result)
    save_results(final_result, llm_result, OUTPUT_PATH, OUTPUT_RECORD_PATH)



if __name__ == "__main__":
    main()
