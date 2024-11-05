import torch
from compute_lip import *
import numpy as np

ckpt = torch.load('models/mnist_4C3F_wd_9.pth','cpu')

items_list = list(ckpt.items())

W = []
N = 1000

for k, v in ckpt.items():
        if 'weight' in k:
            W.append(v)

stride_vec = [(1,1),(2,2),(1,1),(2,2),(),(),()]
padding = (1,1)
input_size_vec = [(1,1,28,28), (1,32,28,28), (1,32,14,14), (1,64,14,14), (1,7*7*64), (1,512), (1,512)]

start_time = time.time()

m = [lipschitz_general(W[0], input_size_vec[0], stride_vec[0], padding, iter = N)]

for kk in range(len(W) - 1):
    H_list = [W[0]]
    strides = [stride_vec[0]]
    for jj in range(kk + 1):
        H_list.append(W[jj + 1]* 0.5)
        strides.append(stride_vec[jj + 1])

    tmp = lipschitz_general(H_list, input_size_vec[0], strides, padding, iter = N)

    sum_val = 0
    for jj in range(kk + 1):
        H2_list = [W[jj + 1]]
        input_size = input_size_vec[jj + 1]
        strides = [stride_vec[jj+1]]
        for ii in range(jj + 2, kk + 2):
            H2_list.append(W[ii] * 0.5)
            strides.append(stride_vec[ii])

        sum_val += lipschitz_general(H2_list, input_size, strides, padding, iter = N) * m[jj]

    m.append(tmp + 0.5 * sum_val)

# Calculate the elapsed time
elapsed_time = time.time() - start_time

# Return the final Lip and elapsed time
Lip = m[-1]
print(Lip)

Lip_triv = 1
for ii in range(len(W)):
    Lip_triv = Lip_triv*lipschitz_general(W[ii], input_size_vec[ii], stride_vec[ii], padding, iter = N)
print(Lip_triv)