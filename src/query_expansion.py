from abc import ABC, abstractmethod
import re
from vllm import SamplingParams
from prompts.prompt_factory import PromptFactory

from utils import build_prompt_query_expander, SEED

class QueryExpander(ABC):

    def __init__(self, llm_model, tokenizer, sampling_params, lang: str, enable_thinking: bool = False):
        self._llm_model = llm_model
        self._tokenizer = tokenizer
        self._sampling_params = sampling_params
        self._enable_thinking = enable_thinking
        self._lang = lang
        self._prompt_factory = PromptFactory()

    def _get_answer(self, text: str) -> str:
        return text.strip().split("\n")[-1].strip()

    @abstractmethod
    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        pass

class QueryRewriter(QueryExpander):

    def __init__(self, llm_model, tokenizer, sampling_params, lang: str, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, lang=lang, enable_thinking=enable_thinking)
        self._system, self._user = self._prompt_factory.get_prompts("query_rewriter", lang=self._lang)

    def __str__(self):
        return f"QueryRewriter(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(self._tokenizer, self._system, self._user, query, enable_thinking=self._enable_thinking) 
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

    def __init__(self, llm_model, tokenizer, sampling_params, lang: str, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, lang=lang, enable_thinking=enable_thinking)
        self._system, self._user = self._prompt_factory.get_prompts("hyde", lang=self._lang)

    def __str__(self):
        return f"HyDEGenerator(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[str]:
        prompt_list = [
            build_prompt_query_expander(self._tokenizer, self._system, self._user, query, enable_thinking=self._enable_thinking) 
            for query in queries
        ]
        batch_outputs = self._llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self._enable_thinking:
            hypothetical_passages = [q + " " + self._get_answer(out.outputs[0].text) for out, q in zip(batch_outputs, queries)]
        else:
            hypothetical_passages = [q + " " + out.outputs[0].text for out, q in zip(batch_outputs, queries)]
        return hypothetical_passages
    
class QueryDescomposition(QueryExpander):

    def __init__(self, llm_model, tokenizer, sampling_params, lang: str, enable_thinking: bool = False):
        super().__init__(llm_model, tokenizer, sampling_params, lang=lang, enable_thinking=enable_thinking)
        self._system, self._user = self._prompt_factory.get_prompts("multiquery", lang=self._lang)

    def _parse_multiquery_output(self, text: str):
        pattern = r'^\[(?:CORE|DECOMP|EXPAND)\]\s*(.+)$'
        queries = re.findall(pattern, text, flags=re.MULTILINE)
        queries = [q.strip() for q in queries if q.strip()]
        return queries

    def __str__(self):
        return f"QueryDescomposition(model={self._llm_model.model.config._name_or_path}, enable_thinking={self._enable_thinking})"

    def expand(self, queries: list[str], lora_request: str = None) -> list[list[str]]:

        prompt_list = [
            build_prompt_query_expander(self._tokenizer, self._system, self._user, query, enable_thinking=self._enable_thinking) 
            for query in queries
        ]
        batch_outputs = self._llm_model.fast_generate(
            prompt_list,
            sampling_params=self._sampling_params,
            lora_request=lora_request,
        )
        if self._enable_thinking:
            multiqueries = [
                self._parse_multiquery_output(self._get_answer(out.outputs[0].text))
                for out in batch_outputs
            ]
        else:
            multiqueries = [
                self._parse_multiquery_output(out.outputs[0].text) 
                for out in batch_outputs
            ]
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

    model_name = "Qwen/Qwen3-1.7B"
    lang = "es"  # "en" or "es"

        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 8192,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = True,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        seed=SEED
    )

    """qe = QueryRewriter(model, tokenizer, sampling_params, lang=lang, enable_thinking=False)
    print(qe)
    rewritten_queries = qe.expand(queries)
    for i, (q, rq) in enumerate(zip(queries, rewritten_queries)):
        print(f"Original Query {i+1}: {q}")
        print(f"Rewritten Query {i+1}: {rq}\n")

    hyde = HyDEGenerator(model, tokenizer, sampling_params, lang=lang, enable_thinking=False)
    print(hyde)
    hyde_passages = hyde.expand(queries)
    for i, (q, hp) in enumerate(zip(queries, hyde_passages)):
        print(f"Original Query {i+1}: {q}")
        print(f"HyDE Passage {i+1}: {hp}\n")"""

    desc = QueryDescomposition(model, tokenizer, sampling_params, lang=lang, enable_thinking=False)
    print(desc)
    multiqueries = desc.expand(queries)
    for i, (q, mq) in enumerate(zip(queries, multiqueries)):
        print(f"Original Query {i+1}: {q}")
        print(f"MultiQuery {i+1}:\n{mq}\n")
        