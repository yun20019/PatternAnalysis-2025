import os
import random
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from modules import SiameseNetwork
from dataset import generate_dataloaders


# Set seed to ensure reproducibility
def set_seed(seed=42):
    """
    Fix all random seeds to ensure reproducibility of results
    across runs and hardware configurations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Generate Positive and Negative Pairs from Triplets
@torch.no_grad()
def collect_pairs(model, loader, device):
    """
    Extract embeddings from triplet batches and compute cosine similarity
    between anchor-positive and anchor-negative pairs.
    """
    model.eval()
    all_scores, all_labels = [], []

    for a, p, n in loader:
        a, p, n = a.to(device), p.to(device), n.to(device)

        # Forward pass to get embeddings
        ea, ep, en = model.forward_triplet(a, p, n)

        # Compute cosine similarity (higher = more similar)
        pos_sim = torch.nn.functional.cosine_similarity(ea, ep)
        neg_sim = torch.nn.functional.cosine_similarity(ea, en)

        # Store positive pair scores and labels
        all_scores.extend(pos_sim.cpu().numpy())
        all_labels.extend(np.ones_like(pos_sim.cpu().numpy()))  # label 1 = same class

        # Store negative pair scores and labels
        all_scores.extend(neg_sim.cpu().numpy())
        all_labels.extend(np.zeros_like(neg_sim.cpu().numpy()))  # label 0 = different class

    return np.array(all_scores), np.array(all_labels)



# Find optimal threshold
def find_best_threshold(scores, labels):
    """
    Compute ROC curve and determine the optimal similarity threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    return thresholds[best_idx], auc(fpr, tpr), fpr, tpr


# Evaluate model performance with the optimal threshold
def evaluate_at_threshold(scores, labels, thr):
    """
    Evaluate classification accuracy and generate confusion matrix
    at a specific similarity threshold.
    """
    preds = (scores >= thr).astype(int)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds)
    return acc, cm


# Main Evaluation
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    IMAGE_DIR = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-image/image"
    CSV_PATH  = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-metadata.csv"

    # Generate train/validation/test dataloaders
    _, val_loader, test_loader = generate_dataloaders(
        IMAGE_DIR, CSV_PATH, batch_size=4, pairs_per_epoch=10000
    )

    model = SiameseNetwork(embedding_dim=512, freeze_backbone=False).to(device)

    ckpt_path = "best_model_b0_triplet.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    print(f"[INFO] Loaded model from {ckpt_path}")


    # Validation
    print("\n[INFO] Collecting validation pairs...")
    val_scores, val_labels = collect_pairs(model, val_loader, device)

    # Find the best similarity threshold
    best_thr, val_auc, fpr, tpr = find_best_threshold(val_scores, val_labels)
    val_acc, cm_val = evaluate_at_threshold(val_scores, val_labels, best_thr)

    print(f"[INFO] Validation AUC: {val_auc:.3f}")
    print(f"[INFO] Best threshold (Youden): {best_thr:.3f}")
    print(f"[INFO] Validation Accuracy: {val_acc:.3f}")

    # Plot ROC curve for validation set
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {val_auc:.3f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_val.png")
    print("[INFO] Saved ROC curve to roc_curve_val.png")

    # Plot confusion matrix for validation set
    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_val,
        display_labels=["Different", "Same"]
    ).plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val.png")
    print("[INFO] Saved confusion matrix to confusion_matrix_val.png")


    # Evaluation on Test Set
    print("\n[INFO] Evaluating on test set...")
    test_scores, test_labels = collect_pairs(model, test_loader, device)

    test_acc, cm_test = evaluate_at_threshold(test_scores, test_labels, best_thr)
    test_auc = auc(*roc_curve(test_labels, test_scores)[:2])

    print(f"[INFO] Test Accuracy: {test_acc:.3f}")
    print(f"[INFO] Test AUC: {test_auc:.3f}")

    # Plot confusion matrix for test set
    plt.figure(figsize=(5, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm_test,
        display_labels=["Different", "Same"]
    ).plot(cmap="Greens", values_format="d")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")
    print("[INFO] Saved confusion matrix to confusion_matrix_test.png")
