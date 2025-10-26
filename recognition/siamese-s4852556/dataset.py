import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_DIR = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
MASK_DIR = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def has_lesion(mask_path):
    # Convert the segmentation mask to grayscale
    mask = Image.open(mask_path).convert('L')
    return np.array(mask).sum() > 0

def build_image_label_list():
    all_images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")]
    image_paths = [os.path.join(IMG_DIR, f) for f in all_images]
    labels = []

    for f in all_images:
        mask_path = os.path.join(MASK_DIR, f.replace(".jpg", "_segmentation.png"))
        labels.append(1 if has_lesion(mask_path) else 0)  # 1 = melanoma, 0 = normal

    return np.array(image_paths), np.array(labels)

def split_dataset(image_paths, labels):
    # Split train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(image_paths)), test_size=0.3, random_state=42, stratify=labels
    )

    # Split train and val from the train_val set
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.1/0.7, random_state=42, stratify=labels[train_val_idx]
    )

    return train_idx, val_idx, test_idx

class SiameseISICDataset(Dataset):
    def __init__(self, image_paths, labels, indices, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx1 = self.indices[idx]
        img1_path = self.image_paths[idx1]
        label1 = self.labels[idx1]

        # Decide positive or negative pair randomly
        if random.random() < 0.5:
            # Positive pair: same class but not the same image
            candidates = np.where((self.labels[self.indices] == label1) &
                                (self.indices != idx1))[0]
            pair_label = 1
        else:
            # Negative pair: different class (exclude itself just in case)
            candidates = np.where((self.labels[self.indices] != label1) &
                                (self.indices != idx1))[0]
            pair_label = 0

        # If no valid candidate is found (rare case), switch pair type
        if len(candidates) == 0:
            if pair_label == 1:
                candidates = np.where((self.labels[self.indices] != label1) &
                                    (self.indices != idx1))[0]
                pair_label = 0
            else:
                candidates = np.where((self.labels[self.indices] == label1) &
                                    (self.indices != idx1))[0]
                pair_label = 1

        # If still empty (shouldn't happen), fallback to self
        if len(candidates) == 0:
            candidates = [idx]

        idx2 = self.indices[random.choice(candidates)]
        img2_path = self.image_paths[idx2]

        # Load and transform
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, pair_label
