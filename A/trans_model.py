import torch
import torch.nn as nn
import timm

def Trans_CNN(num_classes=24):
    # Load pretrained EfficientNet-B0
    model = timm.create_model('efficientnet_b0', pretrained=True)

    # Replace the final classifier layer
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return model
