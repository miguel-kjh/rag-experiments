import json
import torch
from unsloth import FastLanguageModel
from vllm import SamplingParams

from reranker import Reranker
from utils import SEED

# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are an expert evaluator of information relevance.
Your task is to compare a query with a document and determine how well the document answers or satisfies the query.

Analyze the semantic correspondence, contextual alignment, and completeness of the response.
Use the following scoring criteria:

- 1.0 → The document fully satisfies the query, covering all requested information.
- 0.75 → The document is mostly relevant but lacks some details.
- 0.5 → The document is partially or superficially related to the query.
- 0.25 → The document is barely related to the query.
- 0.0 → The document is completely unrelated to the query.

You must return a short reasoning and a JSON object with two fields:
- "reasoning": a brief justification (1–3 sentences) explaining the score.
- "score": a number between 0 and 1 with two decimals.
"""

# =========================
# USER PROMPT TEMPLATE
# =========================
PROMPT = """### Query:
{question}

### Document:
{context}

### Output format:
{{
  "reasoning": "short explanation of the relationship between query and document",
  "score": number between 0 and 1
}}"""

# Retriever and Advanced Filtering Agent
class Rafa(Reranker):
    
    def __init__(
            self, 
            model_name: str, 
            max_seq_length: int = 2048, 
            max_new_tokens: int = 1024, 
            use_chunking: bool = False,
            batch_size: int = 32,
            load_in_4bit: bool = False,
            top_rank: int = 5
        ):


        super().__init__(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            fast_inference=True,
        )
        FastLanguageModel.for_inference(self._model)

        self._use_chunking = use_chunking
        self._top_rank = top_rank
        self._batch_size = batch_size
        self._max_new_tokens = max_new_tokens

        self._sampling_params = SamplingParams(
            temperature = 0,
            max_tokens = self._max_new_tokens,
            seed = SEED,
        )

    def _parse_model_output(self, output_text: str):
        """Intenta extraer el JSON del texto generado."""
        try:
            # Detectar inicio y fin del bloque JSON (por si hay texto adicional)
            start = output_text.find("{")
            end = output_text.rfind("}") + 1
            json_str = output_text[start:end]
            result = json.loads(json_str)
            return result
        except Exception as e:
            return {"reasoning": "", "score": 0.0}
        
    def _build_prompt(self, tokenizer, question: str, context: str) -> str:
        """Builds a chat-style prompt for Unsloth models using tokenizer templates."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": PROMPT.format(context=context, question=question)},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )
    
    def __str__(self):
        return f"Rafa('{self._model_name}', top_rank={self._top_rank})"
    
    def rerank(self, query: str, docs: list):
        # Implementación específica del reranking
        doc_scores = []
        # in batches
        for i in range(0, len(docs), self._batch_size):
            batch_docs = docs[i:i+self._batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            batch_prompts = [
                self._build_prompt(self._tokenizer, question=query, context=text) 
                for text in batch_texts
            ]

            batch_outputs = self._model.fast_generate(
                batch_prompts,
                sampling_params = self._sampling_params,
                lora_request = None,
            )

            batch_texts = [out.outputs[0].text for out in batch_outputs]
            for doc, output_text in zip(batch_docs, batch_texts):
                parsed_output = self._parse_model_output(output_text)
                score = parsed_output["score"]
                reasoning = parsed_output["reasoning"]
                doc_scores.append((doc, score, reasoning))

        # Sort by score descending
        ranked = sorted(doc_scores, key=lambda x: x[1], reverse=True)
        return ranked[:self._top_rank]
    
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
    reranker = Rafa("Qwen/Qwen3-0.6B", max_seq_length=16000, use_chunking=False, load_in_4bit=False)
    ranked_docs = reranker.rerank("What is the capital of France?", docs)
    for doc, score, reasoning in ranked_docs:
        print(f"Score: {score}, Reasoning: {reasoning}\nDocument: {doc.page_content}\n")
