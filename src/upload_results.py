#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sube a Weights & Biases los resultados producidos por tu script de RAG/retrieval.

Para cada carpeta bajo --results-dir que contenga:
  - hparams.json
  - ranking_results.json

crea un run en W&B con:
  - config = hparams.json
  - summary = métricas (ranking_results.json), con prefijo "rank/"
  - artefacto opcional con ambos JSONs

Ejemplo:
  python upload_results_to_wandb.py --project rag-evals --entity mi_org
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List, Optional
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Upload RAG ranking results to Weights & Biases")

    p.add_argument(
        "--results-dir",
        default="results",
        help="Directorio base que contiene las ejecuciones (se recorre recursivamente).",
    )
    p.add_argument(
        "--project",
        required=True,
        help="Nombre del proyecto en Weights & Biases.",
    )
    p.add_argument(
        "--entity",
        default=None,
        help="Entidad/organización en W&B (opcional si tu cuenta por defecto ya vale).",
    )
    p.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Tags para los runs creados (opcional).",
    )
    p.add_argument(
        "--notes",
        default=None,
        help="Notas para el run (opcional).",
    )
    p.add_argument(
        "--artifact",
        action="store_true",
        help="Si se indica, sube hparams.json y ranking_results.json como artefacto.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="No sube nada; solo muestra qué haría.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignora el marcador .wandb_uploaded y vuelve a subir.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Intenta reusar el run_id si existe .wandb_run_id en la carpeta.",
    )
    p.add_argument(
        "--prefix",
        default="rank/",
        help="Prefijo para las métricas en summary (por defecto 'rank/').",
    )
    return p.parse_args()


def find_run_dirs(results_dir: Path) -> Iterable[Path]:
    """Devuelve carpetas que contienen hparams.json y ranking_results.json."""
    for root, dirs, files in os.walk(results_dir):
        files_set = set(files)
        if "hparams.json" in files_set and "ranking_results.json" in files_set:
            yield Path(root)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = "/",
) -> Dict[str, Any]:
    """Aplana un dict anidado (solo dicts/listas sencillas)."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)):
            # Logueamos listas como json string para mantener estructura
            try:
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            except Exception:
                items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def build_names(hparams: Dict[str, Any], run_dir: Path) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Devuelve (run_name, run_group, run_id).
    - name: usa basename de la carpeta; si existe, anexa expid o timestamp
    - group: agrupa por dataset/db/embedding/model si están
    - id: usa timestamp_utc + expid si existen (para --resume)
    """
    base = run_dir.name

    expid = hparams.get("expid")
    ts = hparams.get("timestamp_utc")
    model_alias = hparams.get("model_alias")
    dataset_alias = hparams.get("dataset_alias")
    db_alias = hparams.get("db_alias")
    emb_alias = hparams.get("embedding_alias")

    # Nombre del run
    name_bits = [base]
    if expid:
        name_bits.append(expid)
    run_name = " | ".join(name_bits)

    # Grupo
    group_bits = []
    if dataset_alias:
        group_bits.append(dataset_alias)
    if db_alias:
        group_bits.append(db_alias)
    if emb_alias:
        group_bits.append(emb_alias)
    if model_alias:
        group_bits.append(model_alias)
    run_group = "/".join(group_bits) if group_bits else None

    # ID (para resume/consistencia)
    run_id = None
    if ts and expid:
        run_id = f"{ts}-{expid}"

    return run_name, run_group, run_id


def already_uploaded_marker(run_dir: Path) -> Path:
    return run_dir / ".wandb_uploaded"


def stored_run_id_file(run_dir: Path) -> Path:
    return run_dir / ".wandb_run_id"


def mark_uploaded(run_dir: Path):
    marker = already_uploaded_marker(run_dir)
    try:
        marker.write_text("ok", encoding="utf-8")
    except Exception:
        pass


def store_run_id(run_dir: Path, run_id: str):
    try:
        stored_run_id_file(run_dir).write_text(run_id, encoding="utf-8")
    except Exception:
        pass


def load_stored_run_id(run_dir: Path) -> Optional[str]:
    p = stored_run_id_file(run_dir)
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip() or None
        except Exception:
            return None
    return None


def upload_one_run(
    run_dir: Path,
    project: str,
    entity: Optional[str],
    tags: Optional[List[str]],
    notes: Optional[str],
    prefix: str,
    use_artifact: bool,
    dry_run: bool,
    force: bool,
    resume: bool,
) -> None:
    hparams_path = run_dir / "hparams.json"
    metrics_path = run_dir / "ranking_results.json"

    hparams = read_json(hparams_path)
    metrics = read_json(metrics_path)

    run_name, run_group, run_id_default = build_names(hparams, run_dir)

    # Si hay un run_id guardado, úsalo
    run_id = load_stored_run_id(run_dir) if resume else None
    if resume and not run_id:
        run_id = run_id_default

    # Saltar si ya subido
    if not force and already_uploaded_marker(run_dir).exists():
        print(f"[SKIP] {run_dir} ya marcado como subido. Usa --force para re-subir.")
        return

    print(f"\n=== Subiendo: {run_dir} ===")
    print(f"Run name: {run_name}")
    if run_group:
        print(f"Group:    {run_group}")
    if run_id:
        print(f"Run ID:   {run_id}")

    if dry_run:
        print("[DRY-RUN] No se sube nada. (Mostraríamos config y métricas a continuación)")
        print(f" - Config keys: {list(hparams.keys())[:8]} ...")
        print(f" - Metric keys: {list(metrics.keys())[:8]} ...")
        return

    # Init W&B
    init_kwargs = dict(
        project=project,
        entity=entity,
        name=run_name,
        group=run_group,
        config=hparams,
        tags=tags,
        notes=notes,
        resume="allow" if resume else None,
        id=run_id if resume and run_id else None,
        reinit=True,
    )
    # filtrar None
    init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

    run = wandb.init(**init_kwargs)

    # Guardar run_id por si queremos reanudar más tarde
    if run and run.id:
        store_run_id(run_dir, run.id)

    # Subir métricas a summary (aplanadas y con prefijo)
    flat_metrics = flatten_dict(metrics)
    for k, v in flat_metrics.items():
        wandb.run.summary[f"{prefix}{k}"] = v

    # Guardar archivos como artefacto o adjuntos
    if use_artifact:
        art = wandb.Artifact(
            name=f"{run.name.replace(' ', '_').replace('|','-')}_files",
            type="rag-results",
            metadata={
                "run_dir": str(run_dir),
                "timestamp_utc": hparams.get("timestamp_utc"),
                "expid": hparams.get("expid"),
            },
        )
        art.add_file(str(hparams_path))
        art.add_file(str(metrics_path))
        wandb.log_artifact(art)
    else:
        # Adjuntar directamente al run
        wandb.save(str(hparams_path))
        wandb.save(str(metrics_path))

    wandb.finish()

    # Marcar como subido
    mark_uploaded(run_dir)
    print(f"[OK] Subido {run_dir}")


def main():
    args = parse_args()
    base = Path(args.results_dir).resolve()

    if not base.exists():
        raise SystemExit(f"No existe el directorio: {base}")

    run_dirs = list(find_run_dirs(base))
    if not run_dirs:
        print("No se encontraron carpetas con hparams.json y ranking_results.json.")
        return

    print(f"Encontradas {len(run_dirs)} ejecuciones para subir desde {base}.")

    for rd in sorted(run_dirs):
        try:
            upload_one_run(
                run_dir=rd,
                project=args.project,
                entity=args.entity,
                tags=args.tags,
                notes=args.notes,
                prefix=args.prefix,
                use_artifact=args.artifact,
                dry_run=args.dry_run,
                force=args.force,
                resume=args.resume,
            )
        except Exception as e:
            # Continuamos con las demás, dejando constancia
            print(f"[ERROR] {rd}: {e}")


if __name__ == "__main__":
    main()
