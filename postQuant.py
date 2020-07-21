#########################################################################
# postQuant.py
#
# Dev. Dongwon Paek
# Description: Post-training Quantization using PyTorch
#########################################################################

import os, sys

import torch
import torchvision
import torch.nn as nn

from torch.quantization import QuantStub, DeQuantStub

from models.rexnet import ReXNetV1
from hyperparameter import *

def load_model(model_file):
    model = ReXNetV1(width_mult=0.7)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

def load_dataset_train():
    data_path = '/home/bearpaek/data/datasets/lplSmall/train/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    return train_loader

def load_dataset_test():
    data_path = '/home/bearpaek/data/datasets/lplSmall/validation/'
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    return test_loader

def save_model(model, e):
    model_save_path = './pretrained'
    path = os.path.join(
        model_save_path,
        '{}.pth'.format(e)
    )
    torch.save(model.state_dict(), path)

ORIGINAL_PATH = './pretrained/rexnet_0.7x.pth'

config = CONFIG()
criterion = nn.CrossEntropyLoss()

per_channel_quantized_model = load_model(ORIGINAL_PATH)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
print(per_channel_quantized_model.qconfig)
torch.quantization.prepare(per_channel_quantized_model, inplace=True)
torch.quantization.convert(per_channel_quantized_model, inplace=True)
print('Post Training Quantization: Convert done')

torch.save(per_channel_quantized_model.state_dict(), './pretrained/rexnet_quant.pth')

print("DONE")