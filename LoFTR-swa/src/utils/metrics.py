import torch
import numpy as np
from collections import OrderedDict
from loguru import logger

# --- METRICS ---

@torch.no_grad()
def compute_correspondence_distances(data):
    correspondence_distances = []
    matching_matrix = data['matching_matrix']

    if matching_matrix.dim() != 4 or matching_matrix.size(-1) != 2:
        raise ValueError(f"Unexpected shape for matching_matrix: {matching_matrix.shape}, expected (N, H, W, 2)")

    N, H, W, _ = matching_matrix.shape
    for b in range(N):
        distances = []
        for y in range(H):
            for x in range(W):
                x1, y1 = matching_matrix[b, y, x]
                if x1 > 0 and y1 > 0:
                    dist = torch.norm(torch.tensor([x, y], dtype=torch.float) - torch.tensor([x1, y1], dtype=torch.float))
                    distances.append(dist.item())
        correspondence_distances.append(distances)
    
    # Debugging statements to log type and length
    logger.debug(f"correspondence_distances type: {type(correspondence_distances)}, Length: {len(correspondence_distances)}")
    for idx, dist in enumerate(correspondence_distances):
        logger.debug(f"Type of correspondence_distances[{idx}]: {type(dist)}, Length: {len(dist)}")
    
    assert isinstance(correspondence_distances, list) and all(isinstance(d, list) for d in correspondence_distances), \
        "correspondence_distances should be a list of lists"

    data.update({'correspondence_distances': correspondence_distances})

# --- METRIC AGGREGATION ---

def error_auc(errors, thresholds):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def aggregate_metrics(metrics, distance_thr=5):
    unq_ids = OrderedDict((iden, id) for id, iden in enumerate(metrics['identifiers']))
    unq_ids = list(unq_ids.values())
    logger.info(f'Aggregating metrics over {len(unq_ids)} unique items...')

    distance_thresholds = [5, 10, 20]
    all_correspondence_distances = []

    for i in unq_ids:
        if isinstance(metrics['correspondence_distances'][i], list):
            all_correspondence_distances.extend(metrics['correspondence_distances'][i])
        else:
            logger.warning(f"Unexpected type for correspondence_distances at index {i}: {type(metrics['correspondence_distances'][i])}")

    if not all_correspondence_distances:
        logger.warning("No valid correspondence distances found.")
        return {}

    logger.debug(f"Total correspondence distances: {len(all_correspondence_distances)}")
    correspondence_distances = np.array(all_correspondence_distances)
    aucs = error_auc(correspondence_distances, distance_thresholds)  # (auc@5, auc@10, auc@20)

    return aucs
