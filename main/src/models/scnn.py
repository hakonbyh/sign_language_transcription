import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Identity()

    def forward(self, x):
        xl = list(map(len, x))
        x = torch.cat(x, dim=0)
        features = self.model(x.float())

        return list(features.split(xl))


if __name__ == "__main__":
    efficientnet = ResNet18()
    print(efficientnet.model)
