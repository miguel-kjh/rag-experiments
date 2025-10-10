import json
from unsloth import FastLanguageModel

# Carga del modelo instructivo (puedes cambiarlo por otro)
model_name = "Qwen/Qwen3-1.7B"


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
}}
"""

# =========================
# BUILD PROMPT FUNCTION
# =========================
def build_prompt(tokenizer, question: str, context: str) -> str:
    """Builds a chat-style prompt for Unsloth models using tokenizer templates."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT.format(context=context, question=question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors = "pt",
        return_dict = True,
        enable_thinking=True,
    )

def parse_model_output(output_text: str):
    """Intenta extraer el JSON del texto generado."""
    try:
        # Detectar inicio y fin del bloque JSON (por si hay texto adicional)
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        json_str = output_text[start:end]
        result = json.loads(json_str)
        return result
    except Exception as e:
        print("⚠️ Error parsing JSON:", e)
        return {"reasoning": output_text.strip(), "score": None}
    
# =========================
# EJEMPLO DE USO
# =========================
def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)
    query = "What are the main causes of climate change?"
    document = "Climate change is mainly caused by greenhouse gases like CO2 and methane emitted from human activities."

    input_text = build_prompt(tokenizer, question=query, context=document)
    input_text = input_text.to("cuda")
    outputs = model.generate(**input_text, max_new_tokens=1024)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print(outputs)

    result = parse_model_output(outputs.split("</think>")[-1].strip())
    print("\nParsed output:", result)
# =========================

if __name__ == "__main__":
    main()
