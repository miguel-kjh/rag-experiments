import os
import openai
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate, EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

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
    """Initialize evaluation components (LLM, embeddings)."""
    evaluator_llm = ChatOpenAI(model=MODEL_NAME, seed=SEED, temperature=0.0)
    openai_client = openai.OpenAI()
    embeddings = OpenAIEmbeddings(client=openai_client)
    return evaluator_llm, embeddings


def load_dataset(path: str, subsample: int = None):
    """Load dataset from JSONL and optionally subsample it."""
    dataset = EvaluationDataset.from_jsonl(path)
    if subsample:
        dataset = dataset[:subsample]
    print(f"Loaded dataset with {len(dataset)} samples.")
    return dataset


def run_evaluation(dataset, evaluator_llm):
    """Run evaluation on the dataset using predefined metrics."""
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
    return evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)


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
    result = run_evaluation(dataset, evaluator_llm)
    save_results(result, OUTPUT_PATH)


if __name__ == "__main__":
    main()
