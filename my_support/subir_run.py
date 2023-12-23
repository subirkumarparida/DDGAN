import torch
import os
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])

root_dir = './data/celebahq256_imgs/'
train = os.path.join(root_dir, 'train')
val = os.path.join(root_dir, 'valid')

train_dataset = ImageFolder(root=train, transform=train_transform)
val_dataset = ImageFolder(root=val, transform=train_transform)

print(len(train_dataset), len(val_dataset))
