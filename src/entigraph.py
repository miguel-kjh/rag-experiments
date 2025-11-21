import os
import json
import time
import random
import argparse
from typing import List, Dict, Any, Tuple

from datasets import load_from_disk, load_dataset, Dataset
from tqdm import tqdm

from utils import (
    SEED,
    SYSTEM_PROMPT_FOR_GENERATE_ENTITIES,
    SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS,
    SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS,
)

# ======================================
# Backend único: Unsloth + vLLM
# ======================================

from unsloth import FastLanguageModel
from vllm import SamplingParams


class UnslothBackend:
    """Backend basado en Unsloth + vLLM, con generación batched."""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 4096,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_lora_rank: int = 16,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        print(f"[UnslothBackend] Cargando modelo local: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            load_in_8bit = load_in_8bit,
            fast_inference = True,
            max_lora_rank = max_lora_rank,
        )
        FastLanguageModel.for_inference(self.model)

        self.sampling_params = SamplingParams(
            temperature = temperature,
            max_tokens  = max_tokens,
            seed        = SEED,
        )

        # Si en algún momento quieres usar LoRA:
        # self.lora_request = self.model.load_lora(path)
        self.lora_request = None

    def _build_prompt(self, system_message: str, user_content: str) -> str:
        """Construye el prompt estilo chat para un ejemplo."""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            enable_thinking       = False,
            tokenize              = False,
        )

    def generate_batch(
        self,
        prompts: List[str],
        system_message: str,
    ) -> List[str]:
        """
        prompts: lista de contenidos de usuario (sin system).
        system_message: prompt de sistema común para todos.
        Devuelve una lista de textos generados (uno por prompt).
        """
        if not prompts:
            return []

        batch_prompts = [
            self._build_prompt(system_message, p)
            for p in prompts
        ]
        batch_outputs = self.model.fast_generate(
            batch_prompts,
            sampling_params = self.sampling_params,
            lora_request    = self.lora_request,
        )
        texts = [out.outputs[0].text for out in batch_outputs]
        return texts


# ======================================
# Funciones EntiGraph (versión batched)
# ======================================

def generate_entities_batch(
    backend: UnslothBackend,
    documents: List[str],
    max_retries: int = 5,
    sleep_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """
    Extrae entidades y resumen para una lista de documentos.
    Espera que el modelo devuelva JSON con campos:
      - "summary": str
      - "entities": List[str]
    """
    prompts = [
        f"### Document Content:\n{doc}\n"
        for doc in documents
    ]

    for attempt in range(max_retries):
        try:
            completions = backend.generate_batch(
                prompts = prompts,
                system_message = SYSTEM_PROMPT_FOR_GENERATE_ENTITIES,
            )
            results: List[Dict[str, Any]] = []
            for comp in completions:
                try:
                    resp = json.loads(comp)
                    assert "entities" in resp
                    assert isinstance(resp["entities"], list)
                except Exception as e:
                    print(f"[generate_entities_batch] Error parseando JSON: {e}")
                    resp = {"summary": "", "entities": []}
                results.append(resp)
            return results
        except Exception as e:
            print(f"[generate_entities_batch] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    # Si falla todo, devolvemos “vacíos”
    return [{"summary": "", "entities": []} for _ in documents]


def generate_two_entity_relations_batch(
    backend: UnslothBackend,
    document_content: str,
    entity_pairs: List[Tuple[str, str]],
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> List[str]:
    """
    Genera texto sintético para una lista de pares de entidades
    de un mismo documento.
    """
    prompts = []
    for e1, e2 in entity_pairs:
        prompt = f"""### Document Content:
{document_content}
### Entities:
- {e1}
- {e2}
"""
        prompts.append(prompt)

    if not prompts:
        return []

    for attempt in range(max_retries):
        try:
            completions = backend.generate_batch(
                prompts = prompts,
                system_message = SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS,
            )
            return completions
        except Exception as e:
            print(f"[generate_two_entity_relations_batch] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    return [""] * len(entity_pairs)


def generate_three_entity_relations_batch(
    backend: UnslothBackend,
    document_content: str,
    entity_triples: List[Tuple[str, str, str]],
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> List[str]:
    """
    Genera texto sintético para una lista de tríos de entidades
    de un mismo documento.
    """
    prompts = []
    for e1, e2, e3 in entity_triples:
        prompt = f"""### Document Content:
{document_content}
### Entities:
- {e1}
- {e2}
- {e3}
"""
        prompts.append(prompt)

    if not prompts:
        return []

    for attempt in range(max_retries):
        try:
            completions = backend.generate_batch(
                prompts = prompts,
                system_message = SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS,
            )
            return completions
        except Exception as e:
            print(f"[generate_three_entity_relations_batch] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    return [""] * len(entity_triples)


# ======================================
# Pipeline principal
# ======================================

def build_entigraph_for_dataset(
    backend: UnslothBackend,
    dataset_name: str,
    output_path: str,
    text_column: str = "text",
    max_docs: int = None,
    max_entities_per_doc: int = 30,
    max_pairs_per_doc: int = 50,
    max_triples_per_doc: int = 50,
    batch_size: int = 8,
    seed: int = 42,
) -> None:
    """
    Carga un dataset (Hugging Face Hub o load_from_disk), extrae entidades
    y genera textos sintéticos tipo EntiGraph.
    """
    random.seed(seed)

    print(f"[EntiGraph] Cargando dataset: {dataset_name}")
    ds = load_from_disk(dataset_name)

    if text_column not in ds.column_names:
        raise ValueError(
            f"La columna '{text_column}' no existe en el dataset. "
            f"Columnas disponibles: {ds.column_names}"
        )

    n = len(ds)
    if max_docs is not None:
        n = min(max_docs, n)

    synthetic_records: List[Dict[str, Any]] = []

    for start in tqdm(range(0, n, batch_size), desc="Procesando documentos (batch)"):
        end = min(start + batch_size, n)
        batch_rows   = [ds[i] for i in range(start, end)]
        batch_texts  = [r[text_column] for r in batch_rows]

        # 1) Entidades + resumen EN BATCH
        ent_results = generate_entities_batch(
            backend   = backend,
            documents = batch_texts,
        )

        # Procesamos cada documento del batch
        for local_idx, (row, doc_text, ent_result) in enumerate(
            zip(batch_rows, batch_texts, ent_results)
        ):
            idx = start + local_idx  # índice global

            if not isinstance(doc_text, str) or len(doc_text.strip()) == 0:
                continue

            entities: List[str] = ent_result.get("entities", [])
            summary: str = ent_result.get("summary", "")

            # Limitamos entidades
            if len(entities) > max_entities_per_doc:
                entities = entities[:max_entities_per_doc]

            # Registro de resumen + entidades
            synthetic_records.append(
                {
                    "source_index":   idx,
                    "synthetic_type": "entities_summary",
                    "entities":       entities,
                    "text":           summary,
                }
            )

            # 2) Pares de entidades (batch por doc)
            pair_list: List[Tuple[str, str]] = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    pair_list.append((entities[i], entities[j]))

            if len(pair_list) > max_pairs_per_doc:
                pair_list = random.sample(pair_list, max_pairs_per_doc)

            pair_texts = generate_two_entity_relations_batch(
                backend          = backend,
                document_content = doc_text,
                entity_pairs     = pair_list,
            )

            for (e1, e2), rel_text in zip(pair_list, pair_texts):
                if rel_text:
                    synthetic_records.append(
                        {
                            "source_index":   idx,
                            "synthetic_type": "pair",
                            "entities":       [e1, e2],
                            "text":           rel_text,
                        }
                    )

            # 3) Tríos de entidades (batch por doc)
            triple_list: List[Tuple[str, str, str]] = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    for k in range(j + 1, len(entities)):
                        triple_list.append((entities[i], entities[j], entities[k]))

            random.shuffle(triple_list)
            if len(triple_list) > max_triples_per_doc:
                triple_list = triple_list[:max_triples_per_doc]

            triple_texts = generate_three_entity_relations_batch(
                backend          = backend,
                document_content = doc_text,
                entity_triples   = triple_list,
            )

            for (e1, e2, e3), rel_text in zip(triple_list, triple_texts):
                if rel_text:
                    synthetic_records.append(
                        {
                            "source_index":   idx,
                            "synthetic_type": "triple",
                            "entities":       [e1, e2, e3],
                            "text":           rel_text,
                        }
                    )

    if not synthetic_records:
        print("[EntiGraph] No se generó ningún registro sintético.")
        return

    synth_ds = Dataset.from_list(synthetic_records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jsonl", ".json"]:
        synth_ds.to_json(output_path, lines=(ext == ".jsonl"))
    elif ext in [".parquet"]:
        synth_ds.to_parquet(output_path)
    else:
        synth_ds.to_json(output_path, lines=True)

    print(f"✅ Dataset sintético guardado en: {output_path}")
    print(f"   Número de ejemplos sintéticos: {len(synth_ds)}")


# ======================================
# CLI
# ======================================

def parse_args():
    parser = argparse.ArgumentParser(description="EntiGraph con Unsloth (modelos locales).")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Nombre del dataset en HF Hub o ruta de load_from_disk.",
    )
    parser.add_argument(
        "--local_model_name",
        type=str,
        required=True,
        help="Ruta/nombre del modelo local para Unsloth.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=4096,
        help="Longitud máxima de secuencia.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Cargar modelo en 4bit.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Cargar modelo en 8bit.",
    )
    parser.add_argument(
        "--fast_inference",
        action="store_true",
        help="Activar fast_inference en Unsloth.",
    )
    parser.add_argument(
        "--max_lora_rank",
        type=int,
        default=16,
        help="max_lora_rank para Unsloth.",
    )
    parser.add_argument(
        "--local_temperature",
        type=float,
        default=1.0,
        help="Temperatura para sampling.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Ruta de salida (ej: synthetic_entigraph.jsonl).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Columna de texto del dataset.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Máximo nº de documentos a procesar.",
    )
    parser.add_argument(
        "--max_entities_per_doc",
        type=int,
        default=30,
        help="Máx. entidades por documento.",
    )
    parser.add_argument(
        "--max_pairs_per_doc",
        type=int,
        default=50,
        help="Máx. pares por doc.",
    )
    parser.add_argument(
        "--max_triples_per_doc",
        type=int,
        default=50,
        help="Máx. tríos por doc.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamaño de batch de documentos para extracción de entidades.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla aleatoria.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    backend = UnslothBackend(
        model_name    = args.local_model_name,
        max_seq_length= args.max_seq_length,
        load_in_4bit  = args.load_in_4bit,
        load_in_8bit  = args.load_in_8bit,
        max_lora_rank = args.max_lora_rank,
        temperature   = args.local_temperature,
        max_tokens    = args.local_max_tokens,
    )

    build_entigraph_for_dataset(
        backend              = backend,
        dataset_name         = args.dataset_name,
        output_path          = args.output_path,
        text_column          = args.text_column,
        max_docs             = args.max_docs,
        max_entities_per_doc = args.max_entities_per_doc,
        max_pairs_per_doc    = args.max_pairs_per_doc,
        max_triples_per_doc  = args.max_triples_per_doc,
        batch_size           = args.batch_size,
        seed                 = args.seed,
    )
