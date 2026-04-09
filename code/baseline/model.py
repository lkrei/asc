import torch
import torch.nn as nn
import torchvision.models as models

from config import NUM_CLASSES, MODEL_NAME, PRETRAINED, FREEZE_BACKBONE


class ArchitecturalStyleClassifier(nn.Module):
    
    def __init__(self, num_classes=NUM_CLASSES, model_name=MODEL_NAME, 
                 pretrained=PRETRAINED, freeze_backbone=FREEZE_BACKBONE):
        super(ArchitecturalStyleClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        
        if model_name == "resnet50":
            self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
        elif model_name == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        if self.model_name == "resnet50":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == "efficientnet_b0":
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(num_classes=NUM_CLASSES, model_name=MODEL_NAME, 
                 pretrained=PRETRAINED, freeze_backbone=FREEZE_BACKBONE, 
                 device='cuda'):
    model = ArchitecturalStyleClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    
    model = model.to(device)
    
    trainable, total = model.get_trainable_parameters()
    print(f"модель: {model_name}, обучаемых параметров: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model
