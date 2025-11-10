from datasets import load_from_disk, Dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from unsloth import UnslothTrainer, UnslothTrainingArguments
from trl import SFTTrainer, SFTConfig
import torch

from utils import SEED

FOLDER_KNOWLEDGE = "data/processed/squad_knowledge"
FOLDER_QA = "data/processed/squad_qa"
MODEL_NAME = "meta-llama/Llama-3.2-1B-instruct"
TINY = True
MAX_LENGTH = 2048
SUPER_EPOCHS = 10
RANK_LORA = 8

def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_LENGTH,
        full_finetuning = False,
        load_in_4bit = True,
        load_in_8bit = False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model = FastLanguageModel.get_peft_model(
        model,
        r = RANK_LORA,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = RANK_LORA*2,
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = SEED,
        use_rslora = False,
        loftq_config = None,
    )
    model.print_trainable_parameters()
    return model, tokenizer



def build_prompt_it(tokenizer, system_prompt: str, prompt: str, response: str) -> str:
    """Builds the chat prompt for a single example using the tokenizer chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
        {"role": "assistant", "content": response}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
    )

def generate_qa_prompts(dataset, tokenizer):
    system_prompt = """
    You are an expert at answering questions. Given the question, provide a concise and accurate answer based on your knowledge.
    """
    prompts = []
    for item in dataset:
        prompt = """{QUERY}"""
        response = "{response}"
        question = item["question"] 
        prompt = prompt.format(QUERY=question)
        prompts.append(build_prompt_it(tokenizer, system_prompt, prompt, response.format(response=item["answers"]["text"][0])))
    return prompts

def main():

    knowledge_dataset = load_from_disk(FOLDER_KNOWLEDGE)
    qa_dataset = load_from_disk(FOLDER_QA)
    if TINY:
        print("Using tiny dataset for testing...")
        knowledge_dataset = knowledge_dataset.select(range(128))
        qa_dataset["train"] = qa_dataset["train"].select(range(128))
        qa_dataset["validation"] = qa_dataset["validation"].select(range(64))
    model, tokenizer = get_model()
    
    # datasets
    qa_dataset_train_text = generate_qa_prompts(qa_dataset["train"], tokenizer)
    qa_dataset_validation_text = generate_qa_prompts(qa_dataset["validation"], tokenizer)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    
    qa_train_dataset = Dataset.from_dict({"text": qa_dataset_train_text})
    qa_val_dataset = Dataset.from_dict({"text": qa_dataset_validation_text})

    qa_dataset = {
        "train": qa_train_dataset,
        "validation": qa_val_dataset,
    }
    knowledge_dataset_tokenizer = knowledge_dataset.map(tokenize_function, batched=True)
    qa_train_dataset_tokenizer = qa_dataset["train"].map(tokenize_function, batched=True)
    qa_val_dataset_tokenizer = qa_dataset["validation"].map(tokenize_function, batched=True)
    
    # training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    auto_config = UnslothTrainingArguments(
        per_device_train_batch_size = 8, # TODO: leer la info de unsloth para saber poner esto bien
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        save_strategy="no",
        save_total_limit=0,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        #max_steps = 60,
        learning_rate = 1e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        # 32 bits
        optim = "paged_adamw_32bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = SEED,
        report_to = "none", # Use this for WandB etc
        output_dir="../models/qwen3-0.6b-rag-indexer",
    )

    it_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,        # <-- añade eval batch size
        gradient_accumulation_steps=8,
        warmup_steps=25,
        save_strategy="no",
        save_total_limit=0,
        eval_steps=1,
        eval_strategy="steps",         # <-- activa evaluación periódica
        num_train_epochs=1,             # <-- opcional: usa epochs en lugar de max_steps
        #max_steps=30,
        learning_rate=1e-4,
        logging_steps=1,
        optim = "paged_adamw_32bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        report_to="none",
        output_dir="../models/qwen3-0.6b-rag-retriever",
        load_best_model_at_end=False,          # <-- opcional
        metric_for_best_model="eval_loss",    # <-- opcional
        greater_is_better=False,              # <-- opcional
    )

    trainer_auto = UnslothTrainer(
        model=model,
        train_dataset=knowledge_dataset_tokenizer,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=auto_config,
    )

    trainer_it = SFTTrainer(
        model=model,
        train_dataset=qa_train_dataset_tokenizer,
        eval_dataset=qa_val_dataset_tokenizer,
        data_collator=data_collator,
        tokenizer=tokenizer,
        args=it_config,
    )

    
    for _ in range(SUPER_EPOCHS):
        print(f"--- SUPER EPOCH {_+1} / {SUPER_EPOCHS} ---")
        trainer_sft_stats = trainer_auto.train() 
        trainer_it_stats = trainer_it.train()
        # TODO: guardar modelos intermedios si se quiere y wandb logging etc




if __name__ == "__main__":
    #TODO: pasarlo a hydra
    #TODO: tengo que añadir uno script para evaluar los checkpoints con vllm
    #TODO: AÑADIR LA POSIBILEIDA DE EMPEZAR CON UN CHECKPOINT PREVIO
    main() 
