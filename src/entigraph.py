import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any

from datasets import load_from_disk, load_dataset, Dataset
from tqdm import tqdm

from utils import (
    SEED,
    SYSTEM_PROMPT_FOR_GENERATE_ENTITIES,
    SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS,
    SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS,
)

# -------------------------------------------------------------------
# gptqa: versión standalone basada en el código que me has pasado
# -------------------------------------------------------------------
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


def gptqa(
    prompt: str,
    openai_model_name: str,
    system_message: str,
    json_format: bool = False,
    temp: float = 1.0,
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if json_format:
        completion = client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            seed=SEED,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
    else:
        completion = client.chat.completions.create(
            model=openai_model_name,
            temperature=temp,
            seed=SEED,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
    return completion.choices[0].message.content


# -------------------------------------------------------------------
# Funciones EntiGraph genéricas
# -------------------------------------------------------------------

def generate_entities(
    document_content: str,
    system_message: str,
    openai_model: str,
    max_retries: int = 5,
    sleep_seconds: float = 2.0,
) -> Dict[str, Any]:
    """
    Llama al modelo para extraer:
      - summary
      - entities (lista de strings)
    usando SYSTEM_PROMPT_FOR_GENERATE_ENTITIES.
    """
    prompt = f"""
### Document Content:
{document_content}
"""

    for attempt in range(max_retries):
        try:
            completion = gptqa(
                prompt,
                openai_model,
                system_message,
                json_format=True,
            )
            response = json.loads(completion)
            # Esperamos campos: "summary" y "entities"
            assert "entities" in response
            assert isinstance(response["entities"], list)
            return response
        except Exception as e:
            print(f"[generate_entities] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    raise RuntimeError("Failed to generate entities after several retries.")


def generate_two_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    system_message: str,
    openai_model: str,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    """
    Genera texto sintético para un par de entidades.
    """
    prompt = f"""
### Document Content:
{document_content}
### Entities:
- {entity1}
- {entity2}
"""

    for attempt in range(max_retries):
        try:
            completion = gptqa(
                prompt,
                openai_model,
                system_message,
                json_format=False,
            )
            return completion
        except Exception as e:
            print(f"[generate_two_entity_relations] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    return ""


def generate_three_entity_relations(
    document_content: str,
    entity1: str,
    entity2: str,
    entity3: str,
    system_message: str,
    openai_model: str,
    max_retries: int = 3,
    sleep_seconds: float = 2.0,
) -> str:
    """
    Genera texto sintético para un trío de entidades.
    """
    prompt = f"""
### Document Content:
{document_content}
### Entities:
- {entity1}
- {entity2}
- {entity3}
"""

    for attempt in range(max_retries):
        try:
            completion = gptqa(
                prompt,
                openai_model,
                system_message,
                json_format=False,
            )
            return completion
        except Exception as e:
            print(f"[generate_three_entity_relations] Error intento {attempt + 1}/{max_retries}: {e}")
            time.sleep(sleep_seconds)

    return ""


# -------------------------------------------------------------------
# Pipeline principal para un dataset de HF
# -------------------------------------------------------------------

def build_entigraph_for_dataset(
    dataset_name: str,
    openai_model: str,
    output_path: str,
    text_column: str = "text",
    max_docs: int = None,
    max_entities_per_doc: int = 30,
    max_pairs_per_doc: int = 50,
    max_triples_per_doc: int = 50,
    seed: int = 42,
) -> None:
    """
    Carga un dataset de Hugging Face con una columna `text`, extrae entidades
    y genera corpus sintético con EntiGraph (pares y tríos de entidades).
    Guarda el resultado como un JSONL o similar dependiendo de output_path.
    """
    random.seed(seed)

    print(f"Cargando dataset: {dataset_name}")
    try:
        ds = load_from_disk(dataset_name)
    except Exception:
        ds = load_dataset(dataset_name)

    if text_column not in ds.column_names:
        raise ValueError(
            f"La columna '{text_column}' no existe en el dataset. "
            f"Columnas disponibles: {ds.column_names}"
        )

    if max_docs is not None:
        ds = ds.select(range(min(max_docs, len(ds))))

    synthetic_records: List[Dict[str, Any]] = []

    for idx in tqdm(range(len(ds)), desc="Procesando documentos"):
        row = ds[idx]
        doc_text = row[text_column]

        if not isinstance(doc_text, str) or len(doc_text.strip()) == 0:
            continue

        # 1) Extraer entidades + resumen
        try:
            ent_result = generate_entities(
                document_content=doc_text,
                system_message=SYSTEM_PROMPT_FOR_GENERATE_ENTITIES,
                openai_model=openai_model,
            )
        except Exception as e:
            print(f"[Doc {idx}] Error al generar entidades: {e}")
            continue

        entities: List[str] = ent_result.get("entities", [])
        summary: str = ent_result.get("summary", "")

        # Limitamos número de entidades por documento
        if len(entities) > max_entities_per_doc:
            entities = entities[:max_entities_per_doc]

        # Registro para summary + lista de entidades
        synthetic_records.append(
            {
                "source_index": idx,
                "synthetic_type": "entities_summary",
                "entities": entities,
                "text": summary,
            }
        )

        # 2) Pares de entidades
        pair_list = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair_list.append((entities[i], entities[j]))

        # Submuestreo de pares si hay demasiados
        if len(pair_list) > max_pairs_per_doc:
            pair_list = random.sample(pair_list, max_pairs_per_doc)

        for e1, e2 in pair_list:
            rel_text = generate_two_entity_relations(
                document_content=doc_text,
                entity1=e1,
                entity2=e2,
                system_message=SYSTEM_PROMPT_GENERATE_TWO_ENTITY_RELATIONS,
                openai_model=openai_model,
            )
            if rel_text:
                synthetic_records.append(
                    {
                        "source_index": idx,
                        "synthetic_type": "pair",
                        "entities": [e1, e2],
                        "text": rel_text,
                    }
                )

        # 3) Tríos de entidades
        triple_list = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                for k in range(j + 1, len(entities)):
                    triple_list.append((entities[i], entities[j], entities[k]))

        random.shuffle(triple_list)
        if len(triple_list) > max_triples_per_doc:
            triple_list = triple_list[:max_triples_per_doc]

        for e1, e2, e3 in triple_list:
            rel_text = generate_three_entity_relations(
                document_content=doc_text,
                entity1=e1,
                entity2=e2,
                entity3=e3,
                system_message=SYSTEM_PROMPT_GENERATE_THREE_ENTITY_RELATIONS,
                openai_model=openai_model,
            )
            if rel_text:
                synthetic_records.append(
                    {
                        "source_index": idx,
                        "synthetic_type": "triple",
                        "entities": [e1, e2, e3],
                        "text": rel_text,
                    }
                )

    # Construimos dataset sintético de HF y salvamos
    if not synthetic_records:
        print("No se generó ningún registro sintético.")
        return

    synth_ds = Dataset.from_list(synthetic_records)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardamos según extensión
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jsonl", ".json"]:
        synth_ds.to_json(output_path, lines=(ext == ".jsonl"))
    elif ext in [".parquet"]:
        synth_ds.to_parquet(output_path)
    else:
        # por defecto JSONL
        synth_ds.to_json(output_path, lines=True)

    print(f"✅ Dataset sintético guardado en: {output_path}")
    print(f"Número de ejemplos sintéticos: {len(synth_ds)}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EntiGraph generalizado para HF Datasets.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True, 
        help="Nombre del dataset en Hugging Face Hub (ej: 'ag_news', 'tu_usuario/tu_dataset').",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="Nombre del modelo de OpenAI (por defecto: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Ruta de salida para el dataset sintético (ej: synthetic_entigraph.jsonl).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Nombre de la columna de texto en el dataset (por defecto: text).",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Máximo número de documentos a procesar (por defecto: todos).",
    )
    parser.add_argument(
        "--max_entities_per_doc",
        type=int,
        default=3,
        help="Máximo de entidades por documento (por defecto: 30).",
    )
    parser.add_argument(
        "--max_pairs_per_doc",
        type=int,
        default=5,
        help="Máximo de pares de entidades por documento (por defecto: 50).",
    )
    parser.add_argument(
        "--max_triples_per_doc",
        type=int,
        default=5,
        help="Máximo de tríos de entidades por documento (por defecto: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla aleatoria (por defecto: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_entigraph_for_dataset(
        dataset_name=args.dataset_name,
        openai_model=args.openai_model,
        output_path=args.output_path,
        text_column=args.text_column,
        max_docs=args.max_docs,
        max_entities_per_doc=args.max_entities_per_doc,
        max_pairs_per_doc=args.max_pairs_per_doc,
        max_triples_per_doc=args.max_triples_per_doc,
        seed=args.seed,
    )
