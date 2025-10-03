from typing import List, Sequence, Iterable, Dict, Optional
import numpy as np

# =============== Helpers ===============

def _unique_prefix(seq: Sequence, k: Optional[int] = None):
    """Devuelve los primeros k elementos ÚNICOS en orden."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
            if k is not None and len(out) >= k:
                break
    return out

def _rank_of_relevant_single(pred_ranked: Sequence, gt: Sequence) -> Optional[int]:
    """Rango 1-based del (único) relevante; None si no aparece."""
    gt_set = set(gt)
    r = 0
    seen = set()
    for item in pred_ranked:
        if item in seen:
            continue
        seen.add(item)
        r += 1
        if item in gt_set:
            return r
    return None

def _ranks_from_lists(preds: List[Sequence], gts: List[Sequence]) -> List[Optional[int]]:
    return [_rank_of_relevant_single(p, g) for p, g in zip(preds, gts)]

# =============== Métricas (modo 1 relevante) ===============

def _hitrate_at_k_from_ranks(ranks: List[Optional[int]], k: int) -> float:
    n = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / n if n > 0 else 0.0

def _precision_at_k_from_ranks(ranks: List[Optional[int]], k: int) -> float:
    # Con 1 relevante por query: P@k = Hit@k / k
    return _hitrate_at_k_from_ranks(ranks, k) / k

def _mrr_from_ranks(ranks: List[Optional[int]]) -> float:
    n = len(ranks)
    return (sum((1.0/r) if r is not None else 0.0 for r in ranks) / n) if n > 0 else 0.0

def _cmc_from_ranks(ranks: List[Optional[int]], ks: Iterable[int]) -> Dict[int, float]:
    return {k: _hitrate_at_k_from_ranks(ranks, k) for k in ks}

# =============== Métricas (modo multi-relevante) ===============

def _average_precision_single(pred_ranked: Sequence, gt: Sequence) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    seen = set()
    hits = 0
    precisions = []
    for i, item in enumerate(pred_ranked, start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in gt_set:
            hits += 1
            precisions.append(hits / i)
    return float(np.mean(precisions)) if precisions else 0.0

def _mean_average_precision(preds_ranked: List[Sequence], gts: List[Sequence]) -> float:
    aps = [_average_precision_single(p, g) for p, g in zip(preds_ranked, gts)]
    return float(np.mean(aps)) if aps else 0.0

def _precision_at_k_single_multi(pred_ranked: Sequence, gt: Sequence, k: int) -> float:
    # Divide siempre por k (estándar IR)
    topk = _unique_prefix(pred_ranked, k)
    if not topk:
        return 0.0
    gt_set = set(gt)
    hits = sum(1 for x in topk if x in gt_set)
    return hits / k

def _recall_at_k_single_multi(pred_ranked: Sequence, gt: Sequence, k: int) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    topk = _unique_prefix(pred_ranked, k)
    hits = sum(1 for x in topk if x in gt_set)
    return hits / len(gt_set)

# =============== API principal ===============

def calc_ranking_metrics(
    preds_ranked: List[Sequence],
    gts: List[Sequence],
    ks: Iterable[int] = (1, 2, 3, 5, 10),
    one_relevant_per_query: bool = False,  # True: modo 1-relevante (RAG); False: multi-relevante estándar
    include_classification_view: bool = False,  # opcional: precisión/recall/f1 estilo clasificación (no ranking)
) -> Dict:
    """
    - Si one_relevant_per_query=True: usa el rango del único relevante (recomendado para RAG 1-relevante).
      En este modo: P@k = Recall@k / k,  AP == MRR,  Accuracy@k == Recall@k == CMC@k,
      y F1@k (macro) = Recall@k * 2/(k+1).
    - Si False: usa versión multi-relevante estándar (AP general, P@k/R@k multi).
    """
    assert len(preds_ranked) == len(gts), "preds y gts deben tener la misma longitud"

    ks = sorted(set(int(k) for k in ks if k > 0))
    results: Dict[str, float] = {}

    if one_relevant_per_query:
        ranks = _ranks_from_lists(preds_ranked, gts)

        # MRR y (con 1 relevante) mAP == MRR
        mrr = _mrr_from_ranks(ranks)
        results["MRR"] = mrr
        results["mAP"] = mrr  # igualdad con 1 relevante

        # (Opcional) posición media del relevante
        finite_ranks = [r for r in ranks if r is not None]
        results["AvgRank"] = float(np.mean(finite_ranks)) if finite_ranks else float("inf")

        # CMC/HitRate (== Recall@k y Accuracy@k) y P@k y F1@k
        for k in ks:
            hit = _hitrate_at_k_from_ranks(ranks, k)   # == Recall@k == CMC@k == Accuracy@k
            pk  = _precision_at_k_from_ranks(ranks, k)  # = hit / k
            results[f"CMC@{k}"] = hit
            results[f"Recall@k (macro)@{k}"] = hit
            results[f"Precision@k (macro)@{k}"] = pk
            results[f"Accuracy@{k}"] = hit
            # F1@k (macro) = HitRate@k * 2/(k+1) en 1-relevante
            results[f"F1@k (macro)@{k}"] = hit * (2.0 / (k + 1))

    else:
        # Multi-relevante
        results["mAP"] = _mean_average_precision(preds_ranked, gts)

        # P@k y R@k macro
        for k in ks:
            pks = [_precision_at_k_single_multi(p, g, k) for p, g in zip(preds_ranked, gts)]
            rks = [_recall_at_k_single_multi(p, g, k) for p, g in zip(preds_ranked, gts)]
            results[f"Precision@k (macro)@{k}"] = float(np.mean(pks)) if pks else 0.0
            results[f"Recall@k (macro)@{k}"] = float(np.mean(rks)) if rks else 0.0

        # CMC (hit si aparece cualquiera de los relevantes en top-k únicos)
        n = len(preds_ranked)
        for k in ks:
            hits = 0
            for pred, gt in zip(preds_ranked, gts):
                topk = set(_unique_prefix(pred, k))
                if topk & set(gt):
                    hits += 1
            cmc_k = hits / n if n > 0 else 0.0
            results[f"CMC@{k}"] = cmc_k
            # Accuracy@k (definida como hitrate) para consistencia
            results[f"Accuracy@{k}"] = cmc_k
            # F1@k multi no tiene forma cerrada simple; si la quieres, calcúlala por query.

    # (Opcional) vista "clasificación" global: precision/recall/f1 sobre sets completos (no ranking)
    if include_classification_view:
        vals = []
        for p, g in zip(preds_ranked, gts):
            pred_set, gt_set = set(p), set(g)
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
            vals.append((prec, rec, f1))
        if vals:
            results["classification_precision_macro"] = float(np.mean([v[0] for v in vals]))
            results["classification_recall_macro"]    = float(np.mean([v[1] for v in vals]))
            results["classification_f1_macro"]        = float(np.mean([v[2] for v in vals]))

    return results


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":


    # Lista de predicciones ordenadas (ranking)
    print("Caso de prueba 1:")
    preds = [[1, 2, 3]]
    gts   = [[1, 2, 3]]

    print(calc_ranking_metrics(preds, gts))

    print("Caso de prueba 2:")
    preds = [[1, 2, 3]]
    gts   = [[7, 8, 9]]

    print(calc_ranking_metrics(preds, gts))

    print("Caso de prueba 3:")
    preds = [[10, 20, 30, 40]]
    gts   = [[20, 99]]
    print(calc_ranking_metrics(preds, gts))


    print("Caso de prueba 4:")
    preds = [
        [1772, 1773, 1771, 2722], 
        [4825, 4826, 4828, 4823]
    ]
    gts = [
        [1771, 9999],   # 1771 aparece en preds
        [4823, 8888]    # 4823 aparece en preds
    ]

    print(calc_ranking_metrics(preds, gts))

    print("Caso de prueba 5 (1-relevante):")
    preds = [
        ["1772", "1773", "1771", "2722"],
        ["4825", "4826", "4828", "4823"]
    ]
    gts = [
        ["1772"],   # 1772 aparece en preds
        ["4826"]    # 4825 aparece en preds
    ]
    print(calc_ranking_metrics(preds, gts, one_relevant_per_query=True))

    print("Caso de prueba 6 (1-relevante, no aparece):")
    preds = [
        [1772, 1773, 2722], 
        [4825, 4826, 4828]
    ]
    gts = [
        [1771],   # 1771 NO aparece en preds
        [4823]    # 4823 NO aparece en preds
    ]
    print(calc_ranking_metrics(preds, gts, one_relevant_per_query=True))



