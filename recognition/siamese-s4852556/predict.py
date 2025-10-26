import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from modules import SiameseNetwork
from dataset import SiameseISICDataset, build_image_label_list, split_dataset, transform

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def evaluate(model, test_loader, threshold, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, labels in test_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            out1, out2 = model(x1, x2)
            distances = torch.norm(out1 - out2, p=2, dim=1)
            preds = (distances < threshold).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    return correct / total

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    image_paths, labels = build_image_label_list()
    train_idx, val_idx, test_idx = split_dataset(image_paths, labels)
    test_dataset = SiameseISICDataset(image_paths, labels, test_idx, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load trained model
    model = SiameseNetwork(embedding_dim=128, freeze_backbone=True).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    # Evaluate
    threshold = 1.0 
    acc = evaluate(model, test_loader, threshold, device)
    print(f"Test Accuracy: {acc:.4f}")
