import json
from unsloth import FastLanguageModel


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
    # Carga del modelo instructivo (puedes cambiarlo por otro)
    model_name = "Qwen/Qwen3-0.6B"  
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=2048,
    )
    FastLanguageModel.for_inference(model)
    query = "Por qué el cambio climático es un problema urgente?"
    #document = "El cambio climático es un problema urgente debido a sus efectos devastadores en el medio ambiente, la biodiversidad y las comunidades humanas. El aumento de las temperaturas globales provoca fenómenos meteorológicos extremos, como huracanes, incendios forestales e inundaciones, que amenazan la vida y los medios de subsistencia. Además, el cambio climático contribuye a la pérdida de hábitats naturales y especies, afectando la estabilidad de los ecosistemas. La urgencia radica en la necesidad de mitigar estos impactos mediante la reducción de emisiones de gases de efecto invernadero y la adopción de prácticas sostenibles para proteger nuestro planeta para las generaciones futuras."
    # dame un ejemplo de documento que parezca relevante pero no lo sea
    document = "El calentamiento global es un fenómeno que se refiere al aumento gradual de las temperaturas promedio de la atmósfera terrestre y los océanos. Este proceso ha sido observado durante las últimas décadas y se atribuye principalmente a la actividad humana, como la quema de combustibles fósiles y la deforestación. El calentamiento global tiene diversas consecuencias, incluyendo el derretimiento de los glaciares, el aumento del nivel del mar y cambios en los patrones climáticos. Aunque el calentamiento global es un tema importante, no aborda directamente la urgencia del cambio climático en términos de sus impactos inmediatos y a largo plazo en el medio ambiente y las sociedades humanas."

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
