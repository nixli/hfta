import torch
import torchvision
from torch.utils.data import DataLoader
from hfta.auto_fusion import fuse

import time

def get_fake_dataloader():

    dataset = torchvision.datasets.FakeData(
            size=1000,
            transform=torchvision.transforms.ToTensor()
            )
    loader = DataLoader(dataset, batch_size=8)
    return loader

if __name__ == "__main__":
    models = [
        torchvision.models.resnet18(pretrained=True),
        torchvision.models.resnet50(pretrained=False)
        ]

    fused = fuse.fuse(models)
    fused.cuda()
    loader = get_fake_dataloader()

    print("Staring training")
    start = time.time()
    for d, l in loader:
        d = d.cuda()
        l = l.cuda()
        fused(d)

    print("Done, took {}s".format(time.time() - start))