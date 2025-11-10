from ragas.metrics import (
    StringPresence, 
    ExactMatch,
)
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure
from ragas import evaluate


class Eval:
    pass

class EvalKnowledgeUsingAccuracy(Eval):
    pass

if __name__ == "__main__":
    list_data = [
        {
            "user_input": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "reference": "Paris"
        }
    ]
    evaluation_dataset = EvaluationDataset.from_list(list_data) # Create EvaluationDataset instance
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN), 
            ExactMatch(), 
            StringPresence()
        ]
    )
    print(result)