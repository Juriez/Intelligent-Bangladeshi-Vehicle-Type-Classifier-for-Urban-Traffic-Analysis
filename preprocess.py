import os
import yaml
from sklearn.model_selection import train_test_split 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

data_root = 'data/poribohon-bd'


yaml_path = os.path.join(data_root, 'data.yaml')
with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

classes = data_config.get('names', [])
num_classes = len(classes)
class_to_idx = {name: idx for idx, name in enumerate(classes)}

print(f"Loaded {num_classes} classes from data.yaml: {classes}")


def collect_split(split_name='train'):
    images_dir = os.path.join(data_root, split_name, 'images')
    labels_dir = os.path.join(data_root, split_name, 'labels')

    image_paths = []
    labels = []

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Split '{split_name}' images folder not found: {images_dir}")

    img_files = sorted([f for f in os.listdir(images_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"Processing {len(img_files)} potential images in {split_name}/images")

    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Warning: No label for {img_file} → skipping")
            continue

        with open(label_path, 'r') as lf:
            lines = [line.strip() for line in lf if line.strip()]

        if not lines:
            print(f"Warning: Empty label file {label_file} → skipping")
            continue

     
        first_line_parts = lines[0].split()
        if len(first_line_parts) < 1:
            print(f"Warning: Invalid first line in {label_file} → skipping")
            continue

        try:
            class_id = int(first_line_parts[0])
            if class_id < 0 or class_id >= num_classes:
                print(f"Invalid class_id {class_id} in {label_file} → skipping")
                continue
            labels.append(class_id)
            image_paths.append(img_path)
        except ValueError:
            print(f"Invalid format in {label_file} → skipping")
            continue

    if not image_paths:
        raise ValueError(f"No valid labeled images found in {split_name} split.")

    print(f"{split_name.capitalize()} split: {len(image_paths)} images with labels loaded")
    return image_paths, labels



class PoribohonDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_data_loaders(batch_size=32):
    
    train_paths, train_labels = collect_split('train')
    val_paths, val_labels     = collect_split('valid')
    test_paths, test_labels   = collect_split('test')

    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PoribohonDataset(train_paths, train_labels, train_transform)
    val_dataset   = PoribohonDataset(val_paths,   val_labels,   test_transform)
    test_dataset  = PoribohonDataset(test_paths,  test_labels,  test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

   
    unique, counts = np.unique(train_labels, return_counts=True)
    balance_dict = {classes[i]: cnt for i, cnt in zip(unique, counts)}
    print("Class distribution in train:", balance_dict)

    return train_loader, val_loader, test_loader