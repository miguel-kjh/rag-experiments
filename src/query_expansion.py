from abc import ABC, abstractmethod
from vllm import SamplingParams

from utils import build_prompt_query_expander, SEED

#TODO: usar prompts en español para lo del parlamento

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

    def __init__(self, llm_model, tokenizer, sampling_params, enable_thinking: bool = False):
        self._llm_model = llm_model
        self._tokenizer = tokenizer
        self._sampling_params = sampling_params
        self._enable_thinking = enable_thinking

    def _get_answer(self, text: str) -> str:
        return text.strip().split("\n")[-1].strip()

    @abstractmethod
    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        pass

class QueryRewriter(QueryExpander):

    def __init__(self, llm_model, tokenizer, sampling_params, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, enable_thinking=enable_thinking)

    def __str__(self):
        return f"QueryRewriter(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(self._tokenizer, QUERY_REWRITER_SYSTEM, QUERY_REWRITER_USER, query, enable_thinking=self._enable_thinking) 
            for query in queries
        ]
        batch_outputs = self._llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self._enable_thinking:
            rewritten_queries = [self._get_answer(out.outputs[0].text) for out in batch_outputs]
        else:
            rewritten_queries = [out.outputs[0].text for out in batch_outputs]
        return rewritten_queries
    
class HyDEGenerator(QueryExpander):

    def __init__(self, llm_model, tokenizer, sampling_params, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, enable_thinking=enable_thinking)

    def __str__(self):
        return f"HyDEGenerator(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(self._tokenizer, HYDE_SYSTEM, HYDE_USER, query, enable_thinking=self._enable_thinking) 
            for query in queries
        ]
        batch_outputs = self._llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self._enable_thinking:
            hypothetical_passages = [q + self._get_answer(out.outputs[0].text) for out, q in zip(batch_outputs, queries)]
        else:
            hypothetical_passages = [q + out.outputs[0].text for out, q in zip(batch_outputs, queries)]
        return hypothetical_passages
    

MULTIQUERY_SYSTEM = """You are MultiQuery, a query decomposition and expansion planner for information retrieval.
Goal: from a single user query, generate multiple targeted queries to maximize recall and coverage.
Roles:
- CORE: one rewritten canonical query that preserves intent and is highly retrieval-friendly.
- DECOMP: 2–5 sub-queries that break the task into narrower, complementary aspects.
- EXPAND: 2–5 supporting queries (synonyms, alternate spellings, related entities, acronyms expanded, adjacent topics likely co-mentioned).
Rules:
- Keep the original language of the input.
- Do not change the intent; do not invent facts or specific numbers.
- Add likely clarifications (who/what/when/where), canonical entity names, versions, units, and time ranges when useful.
- Prefer concrete nouns and boolean connectors; include alternate names/spellings where relevant.
- Avoid duplicates and excessive overlap; each query should target a distinct angle.
Output format (one per line), using the role tags for all queries:
[CORE] <query>
[DECOMP] <query>
[EXPAND] <query>"""

# Nota: siguiendo tu preferencia, primero va la query y luego la instrucción.
MULTIQUERY_USER = """Query: {query}
Generate multiple queries as described, listed one per line with their role tags."""
    
class QueryDescomposition(QueryExpander):

    def __init__(self, llm_model, tokenizer, sampling_params, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, enable_thinking=enable_thinking)

    def __str__(self):
        return f"QueryDescomposition(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(self._tokenizer, MULTIQUERY_SYSTEM, MULTIQUERY_USER, query, enable_thinking=self._enable_thinking) 
            for query in queries
        ]
        batch_outputs = self._llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self._enable_thinking:
            multiqueries = [self._get_answer(out.outputs[0].text) for out in batch_outputs]
        else:
            multiqueries = [out.outputs[0].text for out in batch_outputs]
        return multiqueries


if __name__ == "__main__":
    from unsloth import FastLanguageModel
    
    queries = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "How does photosynthesis work?",
        "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?",
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

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        seed=SEED
    )

    qe = QueryRewriter(model, tokenizer, sampling_params, enable_thinking=True)
    print(qe)
    rewritten_queries = qe.expand(queries)
    for i, (q, rq) in enumerate(zip(queries, rewritten_queries)):
        print(f"Original Query {i+1}: {q}")
        print(f"Rewritten Query {i+1}: {rq}\n")

    hyde = HyDEGenerator(model, tokenizer, sampling_params, enable_thinking=True)
    print(hyde)
    hyde_passages = hyde.expand(queries)
    for i, (q, hp) in enumerate(zip(queries, hyde_passages)):
        print(f"Original Query {i+1}: {q}")
        print(f"HyDE Passage {i+1}: {hp}\n")

    desc = QueryDescomposition(model, tokenizer, sampling_params, enable_thinking=False)
    print(desc)
    multiqueries = desc.expand(queries)
    for i, (q, mq) in enumerate(zip(queries, multiqueries)):
        print(f"Original Query {i+1}: {q}")
        print(f"MultiQuery {i+1}:\n{mq}\n")
        