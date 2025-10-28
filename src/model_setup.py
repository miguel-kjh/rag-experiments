from typing import Tuple
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ModelSetup:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_model(
            self,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_target_modules: list = None,
            using_qlora: bool = False,
        ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        pass

class ModelSetupMultiLabelLLM(ModelSetup):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__(model_name)
        self.num_labels = num_labels
    
    def get_model(
            self,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
            lora_target_modules: list = None,
            using_qlora: bool = False
        ) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        

        if using_qlora:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_type="float16"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                quantization_config=bnb_config,
                device_map={"": 0}
            )
            model = prepare_model_for_kbit_training(model)

        else:

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                low_cpu_mem_usage=True,
                device_map={"": 0}
            )
        
        model.config.pad_token_id = tokenizer.pad_token_id        

        config = LoraConfig(
            task_type="SEQ_CLS",
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=lora_target_modules
        )
        model = get_peft_model(model, config)

        for p in model.base_model.model.score.parameters():
            p.requires_grad_(True)

        return model, tokenizer