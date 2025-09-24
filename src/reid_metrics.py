from typing import List, Sequence, Iterable, Tuple, Dict
import numpy as np

def precision_recall_f1_single(pred: Sequence, gt: Sequence):
    pred_set = set(pred)
    gt_set = set(gt)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def precision_recall_f1(preds: List[Sequence], gts: List[Sequence]):
    assert len(preds) == len(gts), "preds and gts must have same length"
    vals = [precision_recall_f1_single(p, g) for p, g in zip(preds, gts)]
    precision = float(np.mean([x[0] for x in vals]))
    recall = float(np.mean([x[1] for x in vals]))
    f1 = float(np.mean([x[2] for x in vals]))
    return {"precision": precision, "recall": recall, "f1": f1}

def average_precision_single(pred_ranked: Sequence, gt: Sequence) -> float:
    gt_set = set(gt)
    if len(gt_set) == 0:
        return 0.0
    seen = set()
    precisions = []
    num_hits = 0
    for i, item in enumerate(pred_ranked, start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in gt_set:
            num_hits += 1
            precisions.append(num_hits / i)
    if len(precisions) == 0:
        return 0.0
    return float(np.mean(precisions))

def mean_average_precision(preds_ranked: List[Sequence], gts: List[Sequence]) -> float:
    assert len(preds_ranked) == len(gts), "preds and gts must have same length"
    aps = [average_precision_single(p, g) for p, g in zip(preds_ranked, gts)]
    return float(np.mean(aps)) if len(aps) > 0 else 0.0

def cmc_curve(preds_ranked: List[Sequence], gts: List[Sequence], max_k: int = None):
    assert len(preds_ranked) == len(gts), "preds and gts must have same length"
    n = len(preds_ranked)
    if n == 0:
        return [], []
    if max_k is None:
        max_k = max(len(p) for p in preds_ranked)
    max_k = max(1, max_k)
    cmc_counts = np.zeros(max_k, dtype=float)
    for pred, gt in zip(preds_ranked, gts):
        gt_set = set(gt)
        hit_rank = None
        seen = set()
        for i, item in enumerate(pred, start=1):
            if item in seen:
                continue
            seen.add(item)
            if item in gt_set:
                hit_rank = i
                break
        if hit_rank is not None and hit_rank <= max_k:
            cmc_counts[hit_rank-1:] += 1.0
    cmc = cmc_counts / n
    ks = list(range(1, max_k+1))
    return ks, cmc.tolist()

def precision_at_k_single(pred_ranked: Sequence, gt: Sequence, k: int) -> float:
    if k <= 0:
        return 0.0
    gt_set = set(gt)
    seen = []
    for item in pred_ranked:
        if item not in seen:
            seen.append(item)
        if len(seen) >= k:
            break
    if len(seen) == 0:
        return 0.0
    hits = sum(1 for x in seen if x in gt_set)
    return hits / len(seen)

def recall_at_k_single(pred_ranked: Sequence, gt: Sequence, k: int) -> float:
    gt_set = set(gt)
    if len(gt_set) == 0:
        return 0.0
    seen = []
    for item in pred_ranked:
        if item not in seen:
            seen.append(item)
        if len(seen) >= k:
            break
    hits = sum(1 for x in seen if x in gt_set)
    return hits / len(gt_set)

def precision_recall_at_k(preds_ranked: List[Sequence], gts: List[Sequence], ks: Iterable[int]):
    assert len(preds_ranked) == len(gts), "preds and gts must have same length"
    rows = []
    for k in ks:
        pks = [precision_at_k_single(p, g, k) for p, g in zip(preds_ranked, gts)]
        rks = [recall_at_k_single(p, g, k) for p, g in zip(preds_ranked, gts)]
        rows.append({"k": k, "Precision@k (macro)": float(np.mean(pks)), "Recall@k (macro)": float(np.mean(rks))})
    return rows


def calc_reid_metrics(preds: List[Sequence], gts: List[Sequence], ks: Iterable[int] = [1, 2, 3, 5, 10]) -> Dict:
    results = {}
    results.update(precision_recall_f1(preds, gts))
    results["mAP"] = mean_average_precision(preds, gts)
    for k, cmc in zip(*cmc_curve(preds, gts, max_k=max(ks))):
        if k in ks:
            results[f"CMC@{k}"] = cmc
    pr_at_k = precision_recall_at_k(preds, gts, ks)
    for row in pr_at_k:
        k = row.pop("k")
        for metric, value in row.items():
            results[f"{metric}@{k}"] = value
    return results

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    def metrics(preds, gts):
        print("Precision, Recall, F1:", precision_recall_f1(preds, gts))
        print("Mean Average Precision:", mean_average_precision(preds, gts))

        ks, cmc = cmc_curve(preds, gts, max_k=5)
        print("CMC Curve:", list(zip(ks, cmc)))

        pr_at_k = precision_recall_at_k(preds, gts, ks=[1, 2, 3])
        for row in pr_at_k:
            print(row)


    # Lista de predicciones ordenadas (ranking)
    print("Caso de prueba 1:")
    preds = [[1, 2, 3]]
    gts   = [[1, 2, 3]]

    metrics(preds, gts)
    print(calc_reid_metrics(preds, gts))

    print("Caso de prueba 2:")
    preds = [[1, 2, 3]]
    gts   = [[7, 8, 9]]

    metrics(preds, gts)

    print("Caso de prueba 3:")
    preds = [[10, 20, 30, 40]]
    gts   = [[20, 99]]
    metrics(preds, gts)


    print("Caso de prueba 4:")
    preds = [
        [1772, 1773, 1771, 2722], 
        [4825, 4826, 4828, 4823]
    ]
    gts = [
        [1771, 9999],   # 1771 aparece en preds
        [4823, 8888]    # 4823 aparece en preds
    ]

    metrics(preds, gts)



