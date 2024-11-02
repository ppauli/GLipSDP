import torch
from compute_lip import *
import numpy as np

ckpt = torch.load('models/mnist_resnet18_fc_wd.pth','cpu')

items_list = list(ckpt.items())

W = []#[ckpt['fc1.weight']]
G = [torch.eye(64)]
H = [torch.zeros(64,28*28)]

for k, v in ckpt.items():
    if 'weight' in k:
        if 'fc1' in k:
            W.append(v)
        if 'fc2' in k:
            G.append(v)
            H.append(torch.eye(64))

W.append(ckpt['fc.weight']) 
#G.append(torch.eye(10))
#H.append(torch.zeros(10,64))

#stride_vec = [(1,1),(2,2),(1,1),(2,2),(),(),()]
#padding = (1,1)
input_size_vec = [(1,28*28)] # use for 2C2F
for ii in range(9):
    input_size_vec.append((1,64))

#input_size_vec = [(1,1,28,28), (1,32,28,28), (1,32,14,14), (1,64,14,14), (1,7*7*64), (1,512), (1,512)]

start_time = time.time()

Hhat = []

for ii in range(len(W)-1):
    Hhat.append(H[ii]+0.5*torch.matmul(G[ii],W[ii]))

m = [lipschitz_general(W[0], input_size_vec[0], iter = 500)]

for kk in range(len(W)-1):
    H_list = []
    #strides = [stride_vec[0]]
    for jj in range(kk + 1):
        H_list.append(Hhat[jj])
        #strides.append(stride_vec[jj + 1])
    H_list.append(W[kk + 1])

    tmp = lipschitz_general(H_list, (1,28*28), iter = 500)

    sum_val = 0
    for jj in range(kk + 1):
        H2_list = [G[jj]]
        input_size = input_size_vec[jj + 1]
        #strides = [stride_vec[jj+1]]
        for ii in range(jj + 1, kk + 1):
            H2_list.append(Hhat[ii])
            #strides.append(stride_vec[ii])
        H2_list.append(W[kk+1])

        sum_val += lipschitz_general(H2_list, input_size, iter = 500) * m[jj]

    m.append(tmp + 0.5 * sum_val)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Return the final Lip and elapsed time
Lip = m[-1]#*lipschitz_general(ckpt['fc.weight'], (10,64), iter = 500)
print(Lip)

Lip_triv = lipschitz_general(W[0], (1,28*28), iter = 500)
for ii in range(1,len(W)-1):
    Lip_triv = Lip_triv*(1+lipschitz_general(G[ii], input_size_vec[ii], iter = 500)*lipschitz_general(W[ii], input_size_vec[ii], iter = 500))
Lip_triv = Lip_triv*lipschitz_general(W[-1], input_size_vec[-1], iter = 500)
print(Lip_triv)