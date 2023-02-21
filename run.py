from __future__ import print_function

import json
import os
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary

from utils import *
from density_plot import get_esd_plot
from models.resnet import resnet
from pyhessian import hessian

import models_vit
#import model_mae_image_loss

import model_mae_image_loss_copy as model_mae_image_loss

# torch.cuda.set_device(2)

# # Scratch
# resume = (
#     "/data/add_disk0/mryoo/tanmay/CVPR2023/output_vit_tiny_scratch/checkpoint-99.pth"
# )

# Our
resume = "/data/add_disk0/mryoo/tanmay/CVPR2023/c_100_ours_new_tiny/checkpoint-99.pth"

# # SSLFT
# resume = (
#     "/data/add_disk0/mryoo/tanmay/CVPR2023/output_vit_tiny_finetune/checkpoint-99.pth"
# )


# Settings
parser = argparse.ArgumentParser(description="PyTorch Example")

parser.add_argument(
    "--mini-hessian-batch-size",
    type=int,
    default=100,
    help="input batch size for mini-hessian batch (default: 200)",
)
parser.add_argument(
    "--hessian-batch-size",
    type=int,
    default=100,
    help="input batch size for hessian (default: 200)",
)

parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")

parser.add_argument("--cuda", action="store_false", help="do we use gpu or not")
parser.add_argument(
    "--resume",
    type=str,
    default=resume,
    help="get the checkpoint",
)

args = parser.parse_args()

# set random seed to reproduce the work
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

for arg in vars(args):
    print(arg, getattr(args, arg))


train_loader, test_loader = getData(
    name="cifar100",
    train_bs=args.mini_hessian_batch_size,
    test_bs=1,
)

assert args.hessian_batch_size % args.mini_hessian_batch_size == 0
assert 50000 % args.hessian_batch_size == 0
batch_num = args.hessian_batch_size // args.mini_hessian_batch_size

# print(type(train_loader))
# print(len(train_loader))

if batch_num == 1:
    for inputs, labels in train_loader:
        hessian_dataloader = (inputs, labels)
        break


else:
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(train_loader):
        hessian_dataloader.append((inputs, labels))
        if i == batch_num - 1:
            break


# print(type(hessian_dataloader[0][0]))
# print(len(hessian_dataloader[0][0]))

# print(type(hessian_dataloader[0]))
# print(len(hessian_dataloader[0]))

# print(type(hessian_dataloader))
# print(len(hessian_dataloader))

# get model


# Scratch

# <class 'torch.Tensor'>
# torch.Size([100, 100])
# t_output: <class 'tuple'>
# <class 'torch.Tensor'>
# <class 'torch.Tensor'>
# gradsh: <class 'list'>
# 152
# <class 'torch.Tensor'>
# torch.Size([1, 1, 192])

# model = models_vit.__dict__["vit_tiny"](
#     num_classes=100,
#     drop_path_rate=0.1,
#     global_pool=True,
#     img_size=32,
#     patch_size=2,
# )

# Ours

# <class 'torch.Tensor'>
# torch.Size([100, 100])
# t_output: <class 'tuple'>
# <class 'torch.Tensor'>
# <class 'float'>
# gradsh: <class 'list'>
# 182
# <class 'torch.Tensor'>
# torch.Size([1, 1, 192])

model = model_mae_image_loss.__dict__["mae_vit_tiny"](
    patch_size=2,
    img_size=32,
    num_classes=100,
    norm_pix_loss=False,
)

# model = model.forward_encoder

# model = models_vit.__dict__["vit_tiny"](
#     num_classes=100,
#     drop_path_rate=0.1,
#     global_pool=True,
#     img_size=32,
#     patch_size=2,
# )

# model = resnet(num_classes=10, depth=20, residual_not=True, batch_norm_not=True)
# <class 'models.resnet.ResNet'>

checkpoint = torch.load(args.resume)
checkpoint_model = checkpoint["model"]
model.load_state_dict(checkpoint_model, strict=False)
# <class 'models_vit.VisionTransformer'>

if args.cuda:
    model = model.cuda()
# model = torch.nn.DataParallel(model)


# model = model.forward_encoder

x = torch.empty(1, 3, 32, 32).cuda()
print(model(x)[1].shape)
# original torch.Size([1, 100])
# only encoder: torch.Size([1, 65, 192])

# print(type(model))
# print(type(model.forward_encoder()))
# summary(model, (3, 32, 32))
# summary(model.forward_encoder, (3, 32, 32))

ssssss

criterion = nn.CrossEntropyLoss()  # label loss

model.eval()
if batch_num == 1:
    hessian_comp = hessian(model, criterion, data=hessian_dataloader, cuda=args.cuda)
else:
    hessian_comp = hessian(
        model, criterion, dataloader=hessian_dataloader, cuda=args.cuda
    )

print("********** finish data londing and begin Hessian computation **********")

# print(torch.cuda.current_device())

top_eigenvalues, _ = hessian_comp.eigenvalues()
trace = hessian_comp.trace()
density_eigen, density_weight = hessian_comp.density()

print("\n***Top Eigenvalues: ", top_eigenvalues)
print("\n***Trace: ", np.mean(trace))

get_esd_plot(density_eigen, density_weight)
