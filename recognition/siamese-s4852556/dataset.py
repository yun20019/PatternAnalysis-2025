import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms



# Dataset for Triplet Loss training
class TripletDataset(Dataset):
    """
    A dataset that creates image triplets for Triplet Loss training.
    """

    def __init__(self, image_dir: str, csv_path: str, transform=None, triplets_per_epoch: int = 20000):
        """
        Initialize dataset by reading metadata and preparing class-wise image lists.
        """
        super().__init__()
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path)

        self.metadata['image_name'] = self.metadata['isic_id'].astype(str) + '.jpg'

        # Separate image names by class
        self.class0 = self.metadata[self.metadata['target'] == 0]['image_name'].tolist()
        self.class1 = self.metadata[self.metadata['target'] == 1]['image_name'].tolist()

        # Augmentations to improve generalization
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.triplets_per_epoch = triplets_per_epoch


    def __len__(self):
        """
        Return the number of triplets to be generated in one epoch.
        """
        return self.triplets_per_epoch


    def __getitem__(self, idx):
        """
        Randomly generate one triplet.
        """
        # Randomly decide which class the anchor belongs to
        anchor_class = 0 if random.random() < 0.5 else 1
        pos_pool = self.class0 if anchor_class == 0 else self.class1
        neg_pool = self.class1 if anchor_class == 0 else self.class0

        # Pick two images of the same class for anchor and positive
        if len(pos_pool) >= 2:
            anchor_name, positive_name = random.sample(pos_pool, 2)
        else:
            # If only one image is available, use it for both anchor and positive
            anchor_name = positive_name = pos_pool[0]

        # Pick one image from the opposite class as negative
        negative_name = random.choice(neg_pool)

        anchor = Image.open(os.path.join(self.image_dir, anchor_name)).convert('RGB')
        positive = Image.open(os.path.join(self.image_dir, positive_name)).convert('RGB')
        negative = Image.open(os.path.join(self.image_dir, negative_name)).convert('RGB')

        # Apply transformations
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative



# Dataloader Generation Function
def generate_dataloaders(image_dir: str, csv_path: str, batch_size: int = 32, pairs_per_epoch: int = 20000):
    """
    Generate dataLoaders for Triplet Loss training, validation, and testing.
    """

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator().manual_seed(seed)

    dataset = TripletDataset(image_dir, csv_path, triplets_per_epoch=pairs_per_epoch)

    # Split dataset: 75% train, 15% validation, 10% test
    train_size = int(len(dataset) * 0.75)
    val_size = int(len(dataset) * 0.15)
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create DataLoaders for each split
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=1, pin_memory=True
    )

    return train_loader, val_loader, test_loader
