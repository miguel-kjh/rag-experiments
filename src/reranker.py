import torch
from abc import ABC, abstractmethod
from sentence_transformers.cross_encoder import CrossEncoder


class Reranker(ABC):
    def __init__(self, model_name: str):
        self._model_name = model_name

    @abstractmethod
    def rerank(self, query: str, docs: list):
        pass

class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, top_rank: int = 5):
        super().__init__(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(model_name, device=device)
        self.top_rank = top_rank

    def __str__(self):
        return f"CrossEncoderReranker('{self._model_name}', top_rank={self.top_rank})"

    def rerank(self, query: str, docs: list):

        pairs = [(query, d.page_content) for d in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return ranked[:self.top_rank]
    

if __name__ == "__main__":
    # Example usage
    from langchain_core.documents import Document

    docs = [
        Document(page_content="La capital de Francia es París", metadata={"id": 1}),
        Document(page_content="Berlin is the capital of Germany.", metadata={"id": 2}),
        Document(page_content="Madrid is the capital of Spain.", metadata={"id": 3}),
        Document(page_content="Rome is the capital of Italy.", metadata={"id": 4}),
        Document(page_content="Lisbon is the capital of Portugal.", metadata={"id": 5}),
        Document(page_content="The capital of France is known for the Eiffel Tower.", metadata={"id": 6}),
    ]

    reranker = CrossEncoderReranker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", top_rank=1)
    query = "What is the capital of France?"
    reranked_docs = reranker.rerank(query, docs)

    for doc, score in reranked_docs:
        print(f"Score: {score:.4f}, Content: {doc}")