import torch
from compute_lip import *
import numpy as np

ckpt = torch.load('models/cifar_LeNet5_wd.pth','cpu')

items_list = list(ckpt.items())
N = 2000

W = []

for k, v in ckpt.items():
        if 'weight' in k:
            W.append(v)

stride_vec = [(1,1),(1,1),(),(),()]
padding = (0,0)
input_size_vec = [(1,3,32,32), (1,6,14,14), (1,400), (1,120), (1,84)]

conv_layer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=padding, bias=False)
conv_layer1.weight = nn.Parameter(W[0])
conv_layer2 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, stride=1, padding=padding, bias=False)
conv_layer2.weight = nn.Parameter(W[1])
fc_layer1 = nn.Linear(400,120)
fc_layer1.weight = nn.Parameter(W[2])
fc_layer2 = nn.Linear(120,84)
fc_layer2.weight = nn.Parameter(W[3])
fc_layer3 = nn.Linear(84,10)
fc_layer3.weight = nn.Parameter(W[4])

Lip_triv = 1
for ii in range(len(W)):
    Lip_triv = Lip_triv*lipschitz_general(W[ii], input_size_vec[ii], stride_vec[ii], padding, iter = N)
print(Lip_triv)

Lip_triv2 = 1
Lip_triv2 = Lip_triv2 * lipschitz_cnn(conv_layer1, input_size_vec[0], iter = N)
Lip_triv2 = Lip_triv2 * lipschitz_cnn(conv_layer2, input_size_vec[1], iter = N)
Lip_triv2 = Lip_triv2 * lipschitz_fc(fc_layer1, input_size_vec[2], iter = N)
Lip_triv2 = Lip_triv2 * lipschitz_fc(fc_layer2, input_size_vec[3], iter = N)
Lip_triv2 = Lip_triv2 * lipschitz_fc(fc_layer3, input_size_vec[4], iter = N)
print(Lip_triv2)
