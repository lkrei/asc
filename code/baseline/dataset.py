import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ArchitecturalStyleDataset(Dataset):
    
    def __init__(self, split_file=None, samples=None, transform=None, data_dir=None):
        self.transform = transform
        self.data_dir = Path(data_dir) if data_dir else None
        
        if split_file is not None:
            with open(split_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.samples = data
            
            split_dir = Path(split_file).parent
            class_to_idx_file = split_dir / "class_to_idx.json"
            idx_to_class_file = split_dir / "idx_to_class.json"
            
            with open(class_to_idx_file, 'r', encoding='utf-8') as f:
                self.class_to_idx = json.load(f)
            
            with open(idx_to_class_file, 'r', encoding='utf-8') as f:
                self.idx_to_class = json.load(f)
        else:
            self.samples = samples
            self.class_to_idx = None
            self.idx_to_class = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['path']
        label = sample['label']
        
        if self.data_dir and not Path(img_path).is_absolute():
            img_path = self.data_dir / img_path
        else:
            img_path = Path(img_path)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(mode='train', image_size=224):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def get_data_loaders(data_splits_file, batch_size=32, num_workers=4, image_size=224, data_dir=None):
    with open(data_splits_file, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)
    
    splits_dir = Path(data_splits_file).parent
    
    train_dataset = ArchitecturalStyleDataset(
        split_file=None,
        samples=splits_data['train'],
        transform=get_transforms('train', image_size),
        data_dir=data_dir
    )
    
    val_dataset = ArchitecturalStyleDataset(
        split_file=None,
        samples=splits_data['val'],
        transform=get_transforms('val', image_size),
        data_dir=data_dir
    )
    
    test_dataset = ArchitecturalStyleDataset(
        split_file=None,
        samples=splits_data['test'],
        transform=get_transforms('test', image_size),
        data_dir=data_dir
    )
    
    class_to_idx_file = splits_dir / "class_to_idx.json"
    idx_to_class_file = splits_dir / "idx_to_class.json"
    
    with open(class_to_idx_file, 'r', encoding='utf-8') as f:
        train_dataset.class_to_idx = json.load(f)
        val_dataset.class_to_idx = train_dataset.class_to_idx
        test_dataset.class_to_idx = train_dataset.class_to_idx
    
    with open(idx_to_class_file, 'r', encoding='utf-8') as f:
        idx_to_class = json.load(f)
        train_dataset.idx_to_class = idx_to_class
        val_dataset.idx_to_class = idx_to_class
        test_dataset.idx_to_class = idx_to_class
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
