from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS


class Retriever(ABC):
    
    def __init__(self, db: FAISS, top_k: int = 5):
        self.db = db
        self.top_k = top_k

    @abstractmethod
    def retrieve(self, query: str):
        pass


class NaiveDenseRetriever(Retriever):


    def __init__(self, db: FAISS, top_k: int = 5, search_type: str = "similarity", lambda_mult: float = 0.5):
        
        assert search_type in ["similarity", "mmr"], "search_type must be 'similarity' or 'mmr'(maximal marginal relevance)"
        assert (search_type == "similarity") or (lambda_mult is not None and 0 <= lambda_mult <= 1), "lambda_mult must be between 0 and 1 when using mmr"
        
        super().__init__(db, top_k=top_k)
        self.search_type = search_type
        self.lambda_mult = lambda_mult
        search_kwargs = {"k": top_k} if search_type == "similarity" else {"k": top_k, "fetch_k": top_k*2, "lambda_mult": lambda_mult}
        self.dense_retriver = self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def __str__(self):
        if self.search_type == "similarity":
            return f"NaiveDenseRetriever(top_k={self.top_k}, search_type='{self.search_type}')"
        else:
            return f"NaiveDenseRetriever(top_k={self.top_k}, search_type='{self.search_type}', lambda_mult={self.lambda_mult})"

    def retrieve(self, query: str):
        return self.dense_retriver.invoke(query)
        