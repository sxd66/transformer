import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def train_trans():
    trans=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    return trans
def test_trans():
    trans=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    ])
    return trans

def dataload(batch_size,val_batch):
    root="../data"
    tra_trans=train_trans()
    val_trans=test_trans()
    train_dataset=datasets.CIFAR100(root,train=True,transform=tra_trans,download=True)
    train_dataload=DataLoader(train_dataset,batch_size,shuffle=True)

    val_dataset = datasets.CIFAR100(root, train=False, transform=val_trans, download=True)
    val_dataload=DataLoader(val_dataset,val_batch)
    return train_dataload,val_dataload




















