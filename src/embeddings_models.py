from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model: str, device: str = 'cuda'):
        self.model = SentenceTransformer(model, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.model.encode_query(text).tolist()