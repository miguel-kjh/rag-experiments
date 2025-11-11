from pyparsing import Any
from ragas.metrics import (
    StringPresence, 
    ExactMatch,
)
from ragas import EvaluationDataset
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure
from ragas import evaluate
from typing import Dict, List, Tuple
import pandas as pd


class Evaluator:
    
    def __init__(self):
        pass

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses should implement this method.")

class EvaluatorKnowledgeUsingAccuracy(Evaluator):
    def __init__(self):
        super().__init__()
        self._metrics = [
            NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN), 
            ExactMatch(), 
            StringPresence()
        ]

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Tuple[Dict[str, float], pd.DataFrame]:

        evaluation_dataset = EvaluationDataset.from_list(dataset) # Create EvaluationDataset instance
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN), 
                ExactMatch(), 
                StringPresence()
            ]
        )

        score_strs = {k: round(v, 4) for k, v in result._repr_dict.items()}

        if "string_present" in score_strs.keys():
            score_strs["accuracy"] = score_strs.pop("string_present")
        if "non_llm_string_similarity" in score_strs.keys():
            score_strs["levenshtein_similarity"] = score_strs.pop("non_llm_string_similarity")

        return score_strs, result.to_pandas()

if __name__ == "__main__":
    list_data = [
        {
            "user_input": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "reference": "Paris"
        },
        {
            "user_input": "What is 2 + 2?",
            "response": "2 + 2 equals 4.",
            "reference": "4"
        }
    ]
    evaluator = EvaluatorKnowledgeUsingAccuracy()
    results = evaluator.evaluate(list_data)
    print(results)