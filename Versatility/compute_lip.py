import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def lipschitz_general(weight, input_size, stride=(1,1), padding=(1,1), iter = 500):

    x = torch.randn(input_size)

    if isinstance(weight, list):
        for _ in range(iter):
            for ii in range(len(weight)):
                if len(weight[ii].shape) == 4 :
                    x = F.conv2d(x, weight[ii], bias=None, stride=stride[ii], padding=padding)
                elif len(weight[ii].shape) == 2 :
                    if len(x.shape) == 4 :
                        originalsize = x.shape
                        x = torch.flatten(x,1)
                    x = F.linear(x, weight[ii])

            for ii in range(len(weight) - 1, -1, -1):
                if len(weight[ii].shape) == 4 :
                    if len(x.shape) == 2:
                        x = x.reshape(originalsize)
                    x = F.conv_transpose2d(x, weight[ii], bias=None, stride=stride[ii], padding=padding)
                    x = F.normalize(x, dim=(0, 1, 2, 3))
                elif len(weight[ii].shape) == 2 :
                    x = F.linear(x, weight[ii].T)
                    x = F.normalize(x)

        for ii in range(len(weight)):
            if len(weight[ii].shape) == 4 :
                x = F.conv2d(x, weight[ii], bias=None, stride=stride[ii], padding=padding)
            elif len(weight[ii].shape) == 2 :
                if len(x.shape) == 4 :
                    x = torch.flatten(x,1)
                x = F.linear(x, weight[ii])
    else:
        for _ in range(iter):
            if len(weight.shape) == 4 :
                x = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
                x = F.conv_transpose2d(x, weight, bias=None, stride=stride, padding=padding)
                x = F.normalize(x, dim=(0, 1, 2, 3))
            if len(weight.shape) == 2 :
                x = F.linear(x, weight)
                x = F.linear(x, weight.T)
                x = F.normalize(x)
        if len(weight.shape) == 4 :
            x = F.conv2d(x,weight,bias=None,stride=stride,padding=padding)
        if len(weight.shape) == 2 :
            x = F.linear(x,weight)

    return x.norm()

def lipschitz_cnn(layer, input_size, iter = 500):

    weight = layer.weight
    stride = layer.stride
    padding = layer.padding

    x = torch.randn(input_size)
    y = layer(x)
    for _ in range(iter):

        y = F.conv_transpose2d(y, weight, bias=None, stride=stride, padding=padding)
        y = F.conv2d(y, weight, bias=None, stride=stride, padding=padding)
        y = F.normalize(y, dim=(1, 2, 3))

    y = F.conv_transpose2d(y, weight, bias=None, stride=stride, padding=padding)
    return y.norm()

def lipschitz_fc(layer, input_size, iter=500):

    weight = layer.weight

    x = torch.randn(input_size)
    x = F.normalize(x)

    for _ in range(iter):
        x = F.linear(x, weight)
        x = F.linear(x, weight.T)
        x = F.normalize(x)

    x = F.linear(x, weight)
    return x.norm()

def lipschitz_res(G_list, W_list, Wkp1, Gj, input_size, stride=(1,1), padding=(1,1), iter = 500):
    x = torch.randn(input_size)
    for _ in range(iter):
        x = F.conv2d(x, Gj, bias=None, stride=stride, padding=padding)
        for ii in range(len(W_list)):
            x = eval_res(x, W_list[ii], G_list[ii], stride=stride, padding=padding)
        x = F.conv2d(x, Wkp1, bias=None, stride=stride, padding=padding)
        x = F.conv_transpose2d(x, Wkp1, bias=None, stride=stride, padding=padding)  
        for ii in range(len(W_list) - 1, -1, -1):
            x = eval_res_transposed(x, W_list[ii], G_list[ii], stride=stride, padding=padding)
        x = F.conv_transpose2d(x, Gj, bias=None, stride=stride, padding=padding)
        x = F.normalize(x, dim=(0,1, 2, 3))
        x = F.conv2d(x, Gj, bias=None, stride=stride, padding=padding)        
        for ii in range(len(W_list)):
            x = eval_res(x, W_list[ii], G_list[ii], stride=stride, padding=padding)
        x = F.conv2d(x, Wkp1, bias=None, stride=stride, padding=padding)
    return x.norm()
        
def eval_res(y, W, G, stride = (1,1), padding = (1,1)):
    u = y
    y = F.conv2d(y, W, bias=None, stride=stride, padding=padding)
    y = 0.5*F.conv2d(y, G, bias=None, stride=stride, padding=padding)
    y += u
    return y

def eval_res_transposed(y, W, G, stride=(1,1), padding=(1,1)):
    u = y
    y = F.conv_transpose2d(y, G, bias=None, stride=stride, padding=padding)
    y = 0.5*F.conv_transpose2d(y, W, bias=None, stride=stride, padding=padding)
    y += u
    return y


def LipLT_FF(W,stride, padding,input_size):

    start_time = time.time()

    m = [lipschitz_general(W[0], input_size, stride, padding, iter = 200)]

    for kk in range(len(W) - 1):
        H_list = [W[0]]
        for jj in range(kk + 1):
            H_list.append(W[jj + 1]* 0.5)

        sum_val = 0
        for jj in range(kk + 1):
            H2_list = [W[jj + 1]]
            for ii in range(jj + 2, kk + 2):
                H2_list.append(W[ii] * 0.5)

            sum_val += lipschitz_general(H2_list, input_size, stride, padding, iter = 200) * m[jj]

        tmp = lipschitz_general(H_list, input_size, stride, padding, iter = 200)
        m.append(tmp + 0.5 * sum_val)

        # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    # Return the final Lip and elapsed time
    Lip = m[-1]

    return Lip, elapsed_time
