import os
import openai
import pandas as pd

from dotenv import load_dotenv
from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from reid_metrics import calc_reid_metrics
from utils import SEED

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "results/generation.jsonl"
OUTPUT_PATH = "results/evaluation.csv"
MODEL_NAME = "gpt-4o"
SUBSAMPLE_SIZE = 5


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
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
    return evaluate(dataset=ragas_dataset, metrics=metrics, llm=evaluator_llm)

def run_reid_evaluation(dataset):
    """Run re-identification metrics evaluation."""
    preds = [item["document_ids"] for item in dataset]
    gts = [item["target_document_ids"] for item in dataset]
    return calc_reid_metrics(preds, gts)


def save_results(result, output_path: str):
    """Convert evaluation results to CSV and save them."""
    print(result)
    df = result.to_pandas()
    df.to_csv(output_path, index=False)
    print(f"✅ Results saved to: {output_path}")


def main():
    setup_environment()
    evaluator_llm, _ = build_components()
    dataset = load_dataset(DATASET_PATH, SUBSAMPLE_SIZE)
    reid_result = run_reid_evaluation(dataset)
    llm_result = run_llm_evaluation(dataset, evaluator_llm)
    score_strs = {k: round(v, 4) for k, v in llm_result._repr_dict.items()}
    # join both results
    final_result = score_strs.copy()
    for k, v in reid_result.items():
        final_result[k] = round(v, 4)
    print(final_result)
    exit()
    save_results(llm_result, OUTPUT_PATH)



if __name__ == "__main__":
    main()
