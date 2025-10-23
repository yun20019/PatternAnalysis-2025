import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import SiameseNetwork
from dataset import SiameseISICDataset, build_image_label_list, split_dataset, transform


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # Contrastive Loss Function
    def forward(self, output1, output2, label):
        distances = torch.norm(output1 - output2, p=2, dim=1)
        loss = (1 - label) * 0.5 * distances.pow(2) + \
               label * 0.5 * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return loss.mean()


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x1, x2, labels in train_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)

        optimizer.zero_grad()
        out1, out2 = model(x1, x2)
        loss = criterion(out1, out2, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x1, x2, labels in val_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.float().to(device)
            out1, out2 = model(x1, x2)
            loss = criterion(out1, out2, labels)
            total_loss += loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    image_paths, labels = build_image_label_list()
    train_idx, val_idx, test_idx = split_dataset(image_paths, labels)

    train_dataset = SiameseISICDataset(image_paths, labels, train_idx, transform=transform)
    val_dataset = SiameseISICDataset(image_paths, labels, val_idx, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model, loss, optimizer
    model = SiameseNetwork(embedding_dim=128, freeze_backbone=True).to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # Training
    epochs = 10
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model at epoch {epoch+1}")
