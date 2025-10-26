import os
import csv
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit


IMG_DIR = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-image/image"
GT_CSV = "/home/Student/s4852556/siamese-s4852556/ISIC2020/train-metadata.csv"

# Transform
default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# build image_paths and labels
def build_image_label_list(img_dir=IMG_DIR, gt_csv=GT_CSV):
    image_paths = []
    labels = []
    with open(gt_csv, "r") as f:
        reader = csv.DictReader(f)
        header = [h.lower() for h in reader.fieldnames]

        if "target" in header:
            label_col = "target"
        elif "mel" in header:
            label_col = "MEL"
        elif "benign_malignant" in header:
            label_col = "benign_malignant"
        else:
            raise ValueError(f"Cannot find label column. Headers: {header}")

        for row in reader:
            img_id = row.get("image_name") or row.get("image") or row.get("isic_id")
            if not img_id:
                continue

            img_path = os.path.join(img_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                continue

            # convert labels to 0 / 1
            if label_col.lower() in ["target", "mel"]:
                label = int(float(row[label_col]))
            else:
                # benign_malignant
                label = 1 if row[label_col].strip().lower() == "malignant" else 0

            image_paths.append(img_path)
            labels.append(label)

    return np.array(image_paths), np.array(labels)


# split train / val / test
def split_dataset(image_paths, labels, seed=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    for train_val_idx, test_idx in sss1.split(image_paths, labels):
        pass

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1/0.7, random_state=seed)
    for train_idx, val_idx in sss2.split(image_paths[train_val_idx], labels[train_val_idx]):
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

    return train_idx, val_idx, test_idx


# Siamese Dataset
class SiameseISICDataset(Dataset):
    def __init__(self, image_paths, labels, indices, transform=None, seed=None):
        self.image_paths = image_paths
        self.labels = labels
        self.indices = np.array(indices)
        self.transform = transform or default_transform

        self.by_class = {}
        for i in self.indices:
            y = int(self.labels[i])
            self.by_class.setdefault(y, []).append(i)
        for y in [0, 1]:
            if y not in self.by_class:
                self.by_class[y] = []

        self._rng = random.Random(seed) if seed is not None else random

    def __len__(self):
        return len(self.indices)

    def _pick_different(self, pool, exclude_idx):
        if not pool:
            return exclude_idx
        if len(pool) == 1 and pool[0] == exclude_idx:
            return exclude_idx
        while True:
            j = self._rng.choice(pool)
            if j != exclude_idx:
                return j

    def __getitem__(self, idx):
        idx1 = int(self.indices[idx])
        img1_path = self.image_paths[idx1]
        y1 = int(self.labels[idx1])

        want_positive = (self._rng.random() < 0.5)

        if want_positive and len(self.by_class[y1]) >= 2:
            idx2 = self._pick_different(self.by_class[y1], exclude_idx=idx1)
            pair_label = 1
        else:
            other_class = 1 - y1
            if len(self.by_class[other_class]) > 0:
                idx2 = self._rng.choice(self.by_class[other_class])
                pair_label = 0
            else:
                idx2 = self._pick_different(self.by_class[y1], exclude_idx=idx1)
                pair_label = 1

        img2_path = self.image_paths[idx2]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, pair_label
