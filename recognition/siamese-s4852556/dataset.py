import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms

class SiameseDataset(Dataset):
    def __init__(self, image_dir: str, csv_path: str, transform=None, pairs_per_epoch: int = 20000):
        super().__init__()
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path)
        self.metadata['image_name'] = self.metadata['isic_id'].astype(str) + '.jpg'

         # Separate samples by class (0 and 1)
        self.class0 = self.metadata[self.metadata['target'] == 0]['image_name'].tolist()
        self.class1 = self.metadata[self.metadata['target'] == 1]['image_name'].tolist()

        # Transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.pairs_per_epoch = pairs_per_epoch

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, idx):
        # Randomly decide whether to generate a positive or negative pair
        same = random.random() < 0.5

        if same:
            cls = 0 if random.random() < 0.5 else 1
            pool = self.class0 if cls == 0 else self.class1
            if len(pool) < 2:
                pool = self.class1 if cls == 0 else self.class0
            a, b = random.sample(pool, 2)
            label = 1
        else:
            a = random.choice(self.class0)
            b = random.choice(self.class1)
            label = 0

        img1 = Image.open(os.path.join(self.image_dir, a)).convert('RGB')
        img2 = Image.open(os.path.join(self.image_dir, b)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def generate_dataloaders(image_dir: str, csv_path: str, batch_size: int = 32, pairs_per_epoch: int = 20000):
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator().manual_seed(seed)

    dataset = SiameseDataset(image_dir, csv_path, pairs_per_epoch=pairs_per_epoch)

    # Split the dataset into train, validation, and test sets
    train_size = int(len(dataset) * 0.75)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader
