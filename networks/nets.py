import torch
from pytorch_pretrained_vit import ViT
import torch.nn as nn
import torch.nn.functional as F
# model = ViT('B_16_imagenet1k', pretrained=True)

# print(model)

class MLP(torch.nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.vit = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=2048)
        self.vit.norm = nn.Identity()
        self.vit.fc = nn.Identity()


        self.mlp1 = MLP(768, 512)


    def forward(self, x):
        z = self.vit(x)
        z = self.mlp1(z)

        z = F.normalize(z, dim=-1)

        return z


