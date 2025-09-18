## coger los datos las preguntas del test
## hacer retrival por cada pregunta
## generar la respuesta con el modelo
## guardar las respuestas en un fichero jsonl

from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import load_from_disk
import torch

from utils import (
    SYSTEM_PROMPT,
    PROMPT,
)

MODEL_GENERATION = "unsloth/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 2048
DATASET = "data/processed/legal-qa-v1"


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_GENERATION,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        #full_finetuning = False, # [NEW!] We have full finetuning now!
        #fast_inference= True,  # Enable fast inference 
    )
    dataset = load_from_disk(DATASET)["test"]
    print(f"Loaded {len(dataset)} samples from {DATASET}")
    query = dataset[0]["question"]
    print(f"Example question: {query}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": PROMPT.format(
                    context="", 
                    question=query
                )
        },
    ]

    query = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        tokenize = False,
    )
    _ = model.generate(
        **tokenizer(query, return_tensors = "pt").to("cuda"),
        max_new_tokens = 512, # Increase for longer outputs!
        temperature = 0.0,
        do_sample = False,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )

if __name__ == "__main__":
    main()