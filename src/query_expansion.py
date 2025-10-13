from abc import ABC, abstractmethod
from vllm import SamplingParams

from utils import build_prompt_query_expander, SEED

QUERY_REWRITER_SYSTEM = """You are QueryRewriter, a retrieval-oriented query reformulator.
Goal: rewrite user queries to be more specific, disambiguated, and retrieval-friendly without changing intent.
Rules:
- Keep the original language of the query.
- Expand abbreviations, resolve pronouns, and add likely constraints (who/what/when/where).
- Prefer concrete nouns, canonical entity names, synonyms, and alternate spellings.
- Add time ranges, units, versions, and boolean connectors if useful.
- Do not invent facts or private data; do not add speculative numbers.
- Remove fluff (greetings, meta).
Output: only the rewritten query (one line, no extra text)."""

QUERY_REWRITER_USER = """Query: {query}
Rewrite this query to be more specific and detailed for document retrieval."""


HYDE_SYSTEM = """You are HyDEGenerator, an expert writer who creates hypothetical answers to guide semantic retrieval.
Goal: write a concise, high-quality passage that a good source could contain, to maximize embedding and retrieval quality.
Style: neutral, factual tone; domain terminology; cohesive paragraph(s).
Constraints:
- Keep the input language.
- No fabricated stats, quotes, or citations (“[1]”, DOI, exact numbers) unless clearly generic.
- Prefer definitions, mechanisms, brief lists, and contextual cues (approximate dates, canonical names).
- Length: 100–500 words.
- Do not reveal reasoning steps or meta information (no “I think”, no “as an AI”).
Output: only the passage."""

HYDE_USER = """Imagine you are an expert writing a detailed explanation on: {query}
Write a single cohesive passage to guide retrieval."""



class QueryExpander(ABC):

    def _get_answer(self, text: str) -> str:
        return text.strip().split("\n")[-1].strip()

    @abstractmethod
    def expand(self, llm_model, tokenizer, queries: list[str], lora_request: str =None) -> list[str]:
        pass

class QueryRewriter(QueryExpander):

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1024, enable_thinking: bool = False):
        super().__init__()
        self._sampling_params = SamplingParams(
            temperature = temperature,
            max_tokens = max_tokens,
            seed = SEED,
        )
        self.enable_thinking = enable_thinking

    def __str__(self):
        return "QueryRewriter"

    def expand(self, llm_model, tokenizer, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(tokenizer, QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER, query, enable_thinking=self.enable_thinking) 
            for query in queries
        ]
        batch_outputs = llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        return [out.outputs[0].text for out in batch_outputs]
    
class HyDEGenerator(QueryExpander):

    def __init__(self, temperature: float = 0.7, max_tokens: int = 256, enable_thinking: bool = False):
        super().__init__()
        self._sampling_params = SamplingParams(
            temperature = temperature,
            max_tokens = max_tokens,
            seed = SEED,
        )
        self.enable_thinking = enable_thinking

    def __str__(self):
        return "HyDEGenerator"

    def expand(self, llm_model, tokenizer, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(tokenizer, HYDE_SYSTEM, HYDE_USER, query, enable_thinking=self.enable_thinking) 
            for query in queries
        ]
        batch_outputs = llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self.enable_thinking:
            hypothetical_passages = [self._get_answer(out.outputs[0].text) for out in batch_outputs]
        else:
            hypothetical_passages = [out.outputs[0].text for out in batch_outputs]
        return hypothetical_passages



if __name__ == "__main__":
    from unsloth import FastLanguageModel
    
    queries = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
        "Cuales son las causas del cambio climatico?",
        "\u00bfQu\u00e9 argumentos expuso el grupo parlamentario que se opuso a la propuesta de modificaci\u00f3n del orden del d\u00eda en la sesi\u00f3n del 26 de septiembre de 2023, que implicaba la convalidaci\u00f3n del decreto relativo al impuesto de sucesiones y donaciones?"
    ]

    model_name = "Qwen/Qwen3-0.6B"

        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 8192,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = True,
    )

    qe = QueryRewriter(temperature=0, max_tokens=1024, enable_thinking=False)
    rewritten_queries = qe.expand(model, tokenizer, queries)
    for i, (q, rq) in enumerate(zip(queries, rewritten_queries)):
        print(f"Original Query {i+1}: {q}")
        print(f"Rewritten Query {i+1}: {rq}\n")
    
    hyde = HyDEGenerator(temperature=0, max_tokens=1024, enable_thinking=False)
    hyde_passages = hyde.expand(model, tokenizer, queries)
    for i, (q, hp) in enumerate(zip(queries, hyde_passages)):
        print(f"Original Query {i+1}: {q}")
        print(f"HyDE Passage {i+1}: {hp}\n")