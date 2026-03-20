import torch.nn as nn
import torchvision.models as models


def get_model():

    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Get number of features
    num_features = model.fc.in_features

    # Replace final layer (5 classes)
    model.fc = nn.Linear(num_features, 5)

    return model