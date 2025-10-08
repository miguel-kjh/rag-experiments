import os
import json
import wandb
from pathlib import Path

def find_result_folders(base_dir: str) -> list:
    """Encuentra carpetas de resultados que contengan ambos archivos esperados."""
    folders = []
    for root, dirs, files in os.walk(base_dir):
        if "hparams.json" in files and "ranking_results.json" in files:
            folders.append(Path(root))
    return folders

def extract_cmc_curve(metrics: dict, prefix: str = "CMC@") -> list:
    """Extrae los valores de CMC@k ordenados por k."""
    cmc = []
    for k in sorted(metrics.keys()):
        if k.startswith(prefix):
            try:
                # Extraer el número k y su valor
                k_val = int(k.split("@")[1])
                cmc.append((k_val, metrics[k]))
            except:
                continue
    # Ordenar por k
    cmc.sort(key=lambda x: x[0])
    return [v for _, v in cmc]

def upload_to_wandb(folder: Path, project: str = "rag-eval", entity: str = None):

    # Cargar archivos
    with open(folder / "hparams.json", "r", encoding="utf-8") as f:
        hparams = json.load(f)

    with open(folder / "ranking_results.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Crear nombre del run que es el nombre del folder
    run_name = "".join(folder.parts[4:])

    # Iniciar W&B run
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=hparams,
        dir=str(folder),
        reinit=True,
    )

    # Extraer curva CMC como lista de floats
    cmc_curve = extract_cmc_curve(metrics)

    # Loggear CMC como gráfico
    wandb.log({"CMC Curve": wandb.plot.line_series(
        xs=list(range(1, len(cmc_curve) + 1)),
        ys=[cmc_curve],
        keys=["CMC@k"],
        title="CMC Curve",
        xname="k"
    )})

    # También loggeamos las métricas individuales
    flat_metrics = {k: v for k, v in metrics.items() if not k.startswith("CMC@")}
    wandb.log(flat_metrics)

    # Subir los archivos como artefacto
    artifact = wandb.Artifact(name=f"{run_name}_artifacts", type="run_files")
    artifact.add_file(str(folder / "hparams.json"))
    artifact.add_file(str(folder / "ranking_results.json"))
    wandb.log_artifact(artifact)

    wandb.finish()
    print(f"Subido a wandb: {folder}")

def main():
    base_dir = "results"  # Cambia esto si lo necesitas
    project = "rag-ranking-eval"
    entity = None  # Tu usuario o equipo en wandb

    result_folders = find_result_folders(base_dir)
    print(f"Se encontraron {len(result_folders)} carpetas con resultados.")

    for folder in result_folders:
        try:
            upload_to_wandb(folder, project=project, entity=entity)
        except Exception as e:
            print(f"Error al subir {folder}: {e}")

if __name__ == "__main__":
    main()
