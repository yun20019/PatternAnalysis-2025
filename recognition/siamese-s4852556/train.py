import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from modules import SiameseNetwork
from dataset import generate_dataloaders

# ------------------------------
# Fix seed
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ------------------------------
# Focal Contrastive Loss
# ------------------------------
class FocalContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, gamma=2.0):
        super().__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, output1, output2, label):
        distances = torch.norm(output1 - output2, p=2, dim=1)
        loss_basic = label * 0.5 * distances.pow(2) + \
                     (1 - label) * 0.5 * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        pt = torch.exp(-loss_basic)
        focal_loss = ((1 - pt) ** self.gamma) * loss_basic
        return focal_loss.mean()

# ------------------------------
# Accuracy calculation
# ------------------------------
def calculate_accuracy(model, loader, threshold, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x1, x2, labels in loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            out1, out2 = model(x1, x2)
            distances = torch.norm(out1 - out2, p=2, dim=1)
            preds = (distances < threshold).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# ------------------------------
# Train / Validate
# ------------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for step, (x1, x2, labels) in enumerate(loader):
        x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)
        optimizer.zero_grad()
        with autocast():
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, (x1, x2, labels) in enumerate(loader):
            x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    IMAGE_DIR = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-image/image"
    CSV_PATH = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-metadata.csv"
    train_loader, val_loader, test_loader = generate_dataloaders(
        IMAGE_DIR, CSV_PATH, batch_size=32, pairs_per_epoch=20000 
    )

    model = SiameseNetwork(embedding_dim=256, freeze_backbone=True).to(device)
    criterion = FocalContrastiveLoss(margin=1.0, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()

    epochs = 15  
    best_val_loss = float('inf')
    best_val_acc = 0.0
    threshold = 1.0

    # Track losses and accuracies
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\nStart training for {epochs} epochs")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        train_acc = calculate_accuracy(model, train_loader, threshold, device)
        val_acc = calculate_accuracy(model, val_loader, threshold, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_b0.pth")
            print(f"Saved best model (val_acc={val_acc:.4f})")

    print(f" Best Val Loss: {best_val_loss:.4f}, Best Val Acc: {best_val_acc:.4f}")

    # ------------------------------
    # Plot Loss & Accuracy
    # ------------------------------
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label="Train Acc")
    plt.plot(epochs_range, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curve.png")
    
