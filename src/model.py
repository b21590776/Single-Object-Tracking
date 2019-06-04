import torch
from torchvision import models
import torch.nn as nn


class ConNet(nn.Module):

    def __init__(self):
        super(ConNet, self).__init__()
        vgg = models.vgg16(pretrained=True)
        for param in vgg.parameters():
            param.requires_grad = False
        self.features = vgg.features
        # average pooling
        self.features.mid = nn.Sequential(nn.AvgPool2d((7, 7)))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # dropout
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # dropout
            nn.Dropout(),
            nn.Linear(1024, 4),
        )

    # forward through the neural net
    def forward(self, x, y):
        x1 = self.features(x)
        x1 = x1.view(x.size(0), 512)
        x2 = self.features(y)
        x2 = x2.view(x.size(0), 512)
        x = torch.cat((x1, x2), 1)
        x = self.classifier(x)
        return x
