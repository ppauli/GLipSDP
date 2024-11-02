import torch
from compute_lip import *
import numpy as np

ckpt = torch.load('models/mnist_resnet18_wd.pth','cpu')

items_list = list(ckpt.items())
N = 100

W = []#[ckpt['fc1.weight']]
G = []#[torch.ones(1,16,1,1)/16]
#H = [torch.zeros(1,16,1,1)]
#W_padding=[(1,1)]
#W_stride=[(1,1)]
#G_padding=[(1,1)]
#G_stride=[(1,1)]

for k, v in ckpt.items():
    if 'weight' in k:
        if 'conv1' in k:
            W.append(v)
        if 'conv2' in k:
            G.append(v)

W = W[1:9]
tmp = torch.zeros(16,16,3,3)
for ii in range(16):
    tmp[ii,ii,1,1]=1
W.append(tmp)

#stride_vec = [(1,1),(2,2),(1,1),(2,2),(),(),()]
#padding = (1,1)
input_size = (1,16,14,14)
#for ii in range(1,9):
#    input_size_vec.append((64,64))
#input_size_vec.append((10,64))

#input_size_vec = [(1,1,28,28), (1,32,28,28), (1,32,14,14), (1,64,14,14), (1,7*7*64), (1,512), (1,512)]

start_time = time.time()

m = [lipschitz_general(W[0], input_size, iter = N)]

for kk in range(len(W)-1):
    G_list = []
    W_list = []
    #strides = [stride_vec[0]]
    for jj in range(kk + 1):
        G_list.append(G[jj])
        W_list.append(W[jj])
        #strides.append(stride_vec[jj + 1])
    Wkp1 = W[kk + 1]
    Gj = torch.zeros(16,16,3,3)
    for ii in range(16):
        Gj[ii,ii,1,1]=1

    tmp = lipschitz_res(G_list, W_list, Wkp1, Gj, input_size, iter = N)

    sum_val = 0
    for jj in range(kk + 1):
        Gj = G[jj]
        G_list = []
        W_list = []
        #input_size = input_size_vec[jj + 1]
        #strides = [stride_vec[jj+1]]
        for ii in range(jj + 1, kk + 1):
            G_list.append(G[ii])
            W_list.append(W[ii])
            #strides.append(stride_vec[ii])
        Wkp1 = W[kk+1]

        sum_val += lipschitz_res(G_list, W_list, Wkp1, Gj, input_size, iter = N) * m[jj]

    m.append(tmp + 0.5 * sum_val)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Return the final Lip and elapsed time
Lip = m[-1]*lipschitz_general(ckpt['fc.weight'], (1,784), iter = N)*lipschitz_general(ckpt['conv1.weight'], (1,1,28,28), iter = N)*0.5
print(Lip)

Lip_triv = lipschitz_general(ckpt['conv1.weight'], (1,1,28,28), iter = N)
for ii in range(len(W)-1):
    tmp = (1+lipschitz_general(G[ii], input_size, iter = N)*lipschitz_general(W[ii], input_size, iter = N))
    print(tmp)
    Lip_triv = Lip_triv*tmp
Lip_triv = Lip_triv*lipschitz_general(ckpt['fc.weight'], (1,784), iter = N)*0.5
print(Lip_triv)