import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    CHECKPOINTS_DIR, MODEL_CHECKPOINT_NAME, METRICS_DIR, RESULTS_DIR,
    CLASS_NAMES, MODEL_NAME, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE
)
from model import create_model
from dataset import get_data_loaders


def load_model(model_path, device='cuda', num_classes=25, model_name=None):
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_name is None:
        model_name = checkpoint.get('model_name', MODEL_NAME)
    
    model = create_model(num_classes=num_classes, model_name=model_name, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, class_names):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'f1_per_class': f1_per_class.tolist(),
        'classification_report': report
    }
    
    return metrics, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'количество'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('истинный класс', fontsize=12)
    plt.xlabel('предсказанный класс', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(metrics, save_dir, model_name):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_to_save = {
        'accuracy': metrics['accuracy'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'f1_per_class': metrics['f1_per_class']
    }
    
    results_path = save_dir / f"test_metrics_{model_name}.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    report_path = save_dir / f"classification_report_{model_name}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(metrics['classification_report'], f, indent=2, ensure_ascii=False)


def print_results(metrics, class_names):
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"macro F1-score: {metrics['macro_f1']:.4f}")
    print(f"weighted F1-score: {metrics['weighted_f1']:.4f}")
    
    print("\nf1-score по классам:")
    for i, (class_name, f1) in enumerate(zip(class_names, metrics['f1_per_class'])):
        print(f"{i+1:2d}. {class_name:40s}: {f1:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = CHECKPOINTS_DIR / MODEL_CHECKPOINT_NAME
    
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', MODEL_NAME)
    
    data_splits_file = RESULTS_DIR / "data_splits.json"
    
    idx_to_class_file = RESULTS_DIR / "idx_to_class.json"
    with open(idx_to_class_file, 'r', encoding='utf-8') as f:
        idx_to_class = json.load(f)
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[str(i)] for i in range(num_classes)]
    
    model = load_model(model_path, device=device, num_classes=num_classes, model_name=model_name)
    
    _, _, test_loader = get_data_loaders(
        data_splits_file,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        image_size=IMAGE_SIZE
    )
    
    y_true, y_pred, y_probs = evaluate_model(model, test_loader, device)
    
    metrics, y_true, y_pred = compute_metrics(y_true, y_pred, class_names)
    
    print(f"mодель: {model_name}")
    print_results(metrics, class_names)
    
    save_results(metrics, METRICS_DIR, model_name)
    
    cm_path = METRICS_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)


if __name__ == "__main__":
    main()
