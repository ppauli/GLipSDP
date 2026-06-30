import argparse
import time

import torch
from compute_lip import *


def load_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="LipLT bound for the 2C2F model used in MODEL_FACTORIES.")
    parser.add_argument("--weights", type=str, default="2C2F_instance1.pt")
    parser.add_argument("--iter", type=int, default=1000)
    args = parser.parse_args()

    ckpt = load_state_dict(args.weights)
    N = args.iter

    # Model:
    # SamePadConv2d(1,16,k=4,s=2) -> ReLU
    # SamePadConv2d(16,32,k=4,s=2) -> ReLU
    # Flatten -> Linear(32*7*7,100) -> ReLU -> Linear(100,10)
    W = [
        ckpt["features.0.conv.weight"],
        ckpt["features.2.conv.weight"],
        ckpt["classifier.1.weight"],
        ckpt["classifier.3.weight"],
    ]

    stride_vec = [(2, 2), (2, 2), (), ()]
    # For k=4, s=2, TensorFlow-style same padding on 28 and 14 gives symmetric padding 1.
    padding = (1, 1)
    input_size_vec = [(1, 1, 28, 28), (1, 16, 14, 14), (1, 32 * 7 * 7), (1, 100)]

    start_time = time.time()

    m = [lipschitz_general(W[0], input_size_vec[0], stride_vec[0], padding, iter=N)]

    for kk in range(len(W) - 1):
        H_list = [W[0]]
        strides = [stride_vec[0]]
        for jj in range(kk + 1):
            H_list.append(W[jj + 1] * 0.5)
            strides.append(stride_vec[jj + 1])

        tmp = lipschitz_general(H_list, input_size_vec[0], strides, padding, iter=N)

        sum_val = 0
        for jj in range(kk + 1):
            H2_list = [W[jj + 1]]
            input_size = input_size_vec[jj + 1]
            strides = [stride_vec[jj + 1]]
            for ii in range(jj + 2, kk + 2):
                H2_list.append(W[ii] * 0.5)
                strides.append(stride_vec[ii])

            sum_val += lipschitz_general(H2_list, input_size, strides, padding, iter=N) * m[jj]

        m.append(tmp + 0.5 * sum_val)

    elapsed_time = time.time() - start_time
    Lip = m[-1]

    Lip_triv = 1
    for ii in range(len(W)):
        Lip_triv *= lipschitz_general(W[ii], input_size_vec[ii], stride_vec[ii], padding, iter=N)

    print(f"LipLT: {Lip}")
    print(f"Trivial product bound: {Lip_triv}")
    print(f"Elapsed time: {elapsed_time:.3f} s")


if __name__ == "__main__":
    main()
