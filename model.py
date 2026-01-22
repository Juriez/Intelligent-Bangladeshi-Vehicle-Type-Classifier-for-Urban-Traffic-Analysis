import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_resnet18(num_classes, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if not pretrained:
        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_custom_cnn(num_classes):
    class VehicleCNN(nn.Module):
        def __init__(self, num_classes):
            super(VehicleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 28 * 28, 512)  # 224 → 112 → 56 → 28
            self.fc2 = nn.Linear(512, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    return VehicleCNN(num_classes)