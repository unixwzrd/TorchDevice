#!/usr/bin/env python

# just to tsee what's going on in TorchDevoce

import torch
import torchdevice

device = torch.device('cuda')
print("Device: ", device)

device = torch.device('mps')
print("Device: ", device)