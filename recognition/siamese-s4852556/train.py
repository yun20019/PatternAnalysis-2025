import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from modules import SiameseNetwork
from dataset import generate_dataloaders



# Set seed to ensure reproducibility
def set_seed(seed=42):
    """
    Fix all random seeds for reproducibility across runs.
    This ensures consistent results between executions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# Triplet Loss function
class TripletLoss(nn.Module):
    """
    Implement the Triplet Margin Loss used in metric learning.
    """
    def __init__(self, margin=0.2):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# Training / Validation Loops
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Train the Siamese network for one epoch using Triplet Loss.
    """
    model.train()
    total_loss = 0.0

    for step, (a, p, n) in enumerate(loader):
        a, p, n = a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            ea, ep, en = model.forward_triplet(a, p, n)
            loss = criterion(ea, ep, en)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Print debug info for the first batch
        if step == 0:
            print(f"[DEBUG] first batch loss={loss.item():.4f}")

    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    """
    Validate the Siamese model on a separate dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for a, p, n in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            ea, ep, en = model.forward_triplet(a, p, n)
            loss = criterion(ea, ep, en)
            total_loss += loss.item()

    return total_loss / len(loader)


# Evaluation
def evaluate_pairwise(model, loader, device, threshold=0.5):
    """
    Evaluate pairwise similarity performance using cosine similarity.
    Generates ROC curve, AUC score, and a confusion matrix
    based on a similarity threshold.
    """
    model.eval()
    all_distances, all_labels = [], []

    with torch.no_grad():
        for x1, x2, labels in loader:
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2 = model(x1, x2)

            # Compute cosine similarity between embeddings
            cos_sim = nn.functional.cosine_similarity(emb1, emb2)

            all_distances.extend(cos_sim.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_distances)
    roc_auc = auc(fpr, tpr)

    # Convert similarity scores to binary predictions
    preds = (all_distances > threshold).astype(int)
    cm = confusion_matrix(all_labels, preds)

    return all_labels, all_distances, roc_auc, fpr, tpr, cm



# Main Training 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    IMAGE_DIR = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-image/image"
    CSV_PATH = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-metadata.csv"

    # Initialize dataloaders
    train_loader, val_loader, test_loader = generate_dataloaders(
        IMAGE_DIR, CSV_PATH, batch_size=4, pairs_per_epoch=30000
    )

    # Initialize model, loss, optimizer, and scaler
    model = SiameseNetwork(embedding_dim=512, freeze_backbone=False).to(device)
    criterion = TripletLoss(margin=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scaler = GradScaler()

    # Training configuration
    epochs = 15
    best_val_loss = float("inf")

    print(f"\n[INFO] Start Triplet training for {epochs} epochs")
    print("=" * 60)

    train_losses, val_losses = [], []

    # Epoch Loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # Training and validation for this epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f}")

        # Save best model (lowest validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_b0_triplet.pth")
            print(f"[INFO] Saved best model (val_loss={val_loss:.4f})")

    print(f"\n[INFO] Training finished. Best Val Loss: {best_val_loss:.4f}")

    # Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Triplet Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve_triplet_b0.png")
    print("[INFO] Saved training curve to training_curve_triplet_b0.png")


    print("\n[INFO] Evaluating ROC and Confusion Matrix on validation set...")
    labels, sims, roc_auc, fpr, tpr, cm = evaluate_pairwise(model, val_loader, device, threshold=0.5)

    # Plot ROC Curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Validation Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_val.png")
    print(f"[INFO] ROC AUC = {roc_auc:.3f}")
    print("[INFO] Saved ROC curve to roc_curve_val.png")

    # Plot Confusion Matrix
    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_val.png")
    print("[INFO] Saved confusion matrix to confusion_matrix_val.png")
