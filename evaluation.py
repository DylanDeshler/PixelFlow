"""
Frozen-feature evaluation helpers shared by pretraining (periodic k-NN) and the final eval script.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_features(backbone, loader, device, ctx, desc='features'):
    """
    Global embeddings (backbone(x), the cls token for ViT), L2-normalized, as fp32 on device.
    """
    was_training = backbone.training
    backbone.eval()
    features, labels = [], []
    for x, y in tqdm(loader, desc=desc):
        x = x.to(device, non_blocking=True)
        with ctx:
            f = backbone(x)
        features.append(F.normalize(f.float(), dim=-1))
        labels.append(y.to(device, non_blocking=True))
    backbone.train(was_training)
    return torch.cat(features), torch.cat(labels)

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k=20, T=0.07, num_classes=1000, chunk_size=256):
    """
    Weighted k-NN vote from DINO (Wu et al. 2018): neighbours vote with weight exp(cos_sim / T).
    Returns (top1, top5) in percent.
    """
    top1, top5, total = 0, 0, 0
    for idx in range(0, len(test_features), chunk_size):
        features = test_features[idx:idx + chunk_size]
        targets = test_labels[idx:idx + chunk_size]
        B = targets.shape[0]

        similarity = features @ train_features.T
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        retrieved_neighbors = train_labels[indices]

        retrieval_one_hot = F.one_hot(retrieved_neighbors, num_classes).float()
        distances_transform = (distances / T).exp()
        probs = (retrieval_one_hot * distances_transform.unsqueeze(-1)).sum(1)
        predictions = probs.topk(5, dim=1).indices

        correct = predictions.eq(targets.view(-1, 1))
        top1 += correct[:, :1].sum().item()
        top5 += correct[:, :5].sum().item()
        total += B
    return 100.0 * top1 / total, 100.0 * top5 / total
