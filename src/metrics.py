from __future__ import annotations

import math

import numpy as np


def retrieval_metrics_from_similarity(
    sim: np.ndarray,
    query_ids: np.ndarray,
    candidate_ids: np.ndarray,
    true_candidate_for_query: dict[int, int],
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[str, float]:
    """
    Compute A-to-B retrieval metrics from a similarity matrix.

    sim shape:
        (n_queries, n_candidates)
    """
    n_queries, n_candidates = sim.shape

    if n_queries != len(query_ids):
        raise ValueError("sim row count must match query_ids length.")

    if n_candidates != len(candidate_ids):
        raise ValueError("sim column count must match candidate_ids length.")

    max_k = min(max(k_values), n_candidates)

    top_part = np.argpartition(sim, -max_k, axis=1)[:, -max_k:]
    top_scores = sim[np.arange(n_queries)[:, None], top_part]
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_sorted = top_part[np.arange(n_queries)[:, None], order]

    true_candidates = np.asarray(
        [true_candidate_for_query[int(qid)] for qid in query_ids],
        dtype=np.int64,
    )

    metrics: dict[str, float] = {}

    for k_req in k_values:
        k = min(k_req, n_candidates)
        retrieved_ids = candidate_ids[top_sorted[:, :k]]
        hit = (retrieved_ids == true_candidates[:, None]).any(axis=1)

        recall = float(hit.mean())
        random_recall = float(min(k / n_candidates, 1.0))

        metrics[f"recall@{k_req}"] = recall
        metrics[f"lift@{k_req}"] = (
            float(recall / random_recall) if random_recall > 0 else math.inf
        )

    candidate_to_idx = {int(cid): idx for idx, cid in enumerate(candidate_ids)}
    true_idx = np.asarray(
        [candidate_to_idx[int(cid)] for cid in true_candidates],
        dtype=np.int64,
    )

    pos_sim = sim[np.arange(n_queries), true_idx]
    metrics["pos_sim_mean"] = float(pos_sim.mean())
    metrics["n_pool"] = float(n_candidates)

    return metrics


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = normalize_rows(a.astype(np.float32))
    b_norm = normalize_rows(b.astype(np.float32))
    return a_norm @ b_norm.T