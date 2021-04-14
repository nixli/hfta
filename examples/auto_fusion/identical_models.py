import torch
import torchvision
from torch.utils.data import DataLoader
from hfta.auto_fusion import fuse

import time

def get_fake_dataloader(batch_size):

    dataset = torchvision.datasets.FakeData(
            size=1000,
            transform=torchvision.transforms.ToTensor()
            )
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader

if __name__ == "__main__":
    models = [
        torchvision.models.resnet18(pretrained=True),
        torchvision.models.resnet18(pretrained=True),
        torchvision.models.resnet18(pretrained=True),
        torchvision.models.resnet18(pretrained=True),
        torchvision.models.resnet18(pretrained=False)
        ]

    batch_size = 8
    fused = fuse.fuse(models)
    fused.cuda()
    loader = get_fake_dataloader(batch_size)
    B = len(models)
    print("Staring training")
    start = time.time()
    for data, label in loader:
        data = data.cuda()

        data = data.unsqueeze(1).expand(-1, B, -1, -1, -1).contiguous()
        label = label.cuda()

        # Do Fusion
        pred = fused(data)

    print("Done, took {}s".format(time.time() - start))