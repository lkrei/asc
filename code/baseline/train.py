import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
from pathlib import Path
import json
from tqdm import tqdm

from config import (
    LEARNING_RATE, NUM_EPOCHS, WEIGHT_DECAY, MOMENTUM,
    CHECKPOINTS_DIR, MODEL_CHECKPOINT_NAME, SAVE_BEST_MODEL, SAVE_LAST_MODEL,
    RESULTS_DIR, METRICS_DIR, MODEL_NAME,
    BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE
)
from model import create_model
from dataset import get_data_loaders


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        
        losses.update(loss.item(), images.size(0))
        accuracies.update(accuracy, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device):
    model.eval()
    
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            
            losses.update(loss.item(), images.size(0))
            accuracies.update(accuracy, images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
    
    return losses.avg, accuracies.avg


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, 
                learning_rate=LEARNING_RATE, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"устройство: {device}, эпох: {num_epochs}, lr: {learning_rate}")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if SAVE_BEST_MODEL:
                checkpoint_name = f"best_model_{MODEL_NAME}.pth"
                checkpoint_path = CHECKPOINTS_DIR / checkpoint_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': history,
                    'model_name': MODEL_NAME
                }, checkpoint_path)
        
        if SAVE_LAST_MODEL:
            last_model_name = f"last_model_{MODEL_NAME}.pth"
            last_model_path = CHECKPOINTS_DIR / last_model_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history,
                'model_name': MODEL_NAME
            }, last_model_path)
    
    elapsed_time = time.time() - start_time
    print(f"лучшая: {best_val_acc:.4f}, время: {elapsed_time/60:.2f}")
    
    history_path = METRICS_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data_splits_file = RESULTS_DIR / "data_splits.json"
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data_splits_file,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")
    
    model = create_model(device=device)
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )


if __name__ == "__main__":
    main()
