import argparse
import time

import torch
from compute_lip import *


def load_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    return ckpt


def conv_identity(channels: int, kernel_size: int = 3):
    eye = torch.zeros(channels, channels, kernel_size, kernel_size)
    center = kernel_size // 2
    for i in range(channels):
        eye[i, i, center, center] = 1
    return eye


def main():
    parser = argparse.ArgumentParser(description="LipLT bound for the C-R18 model used in MODEL_FACTORIES.")
    parser.add_argument("--weights", type=str, default="C-R18_instance3.pt")
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--iter", type=int, default=100)
    args = parser.parse_args()

    ckpt = load_state_dict(args.weights)
    N = args.iter
    num_blocks = args.num_blocks

    # Model:
    # SamePadConv2d(1,16,k=7,s=1) -> ReLU -> MaxPool2d(2,2)
    # blocks.i: conv1 -> ReLU -> conv2 -> + identity -> ReLU
    # AvgPool2d(2,2) -> Flatten -> classifier
    W = [ckpt[f"blocks.{i}.conv1.conv.weight"] for i in range(num_blocks)]
    G = [ckpt[f"blocks.{i}.conv2.conv.weight"] for i in range(num_blocks)]

    # Terminal identity used by the LipLT recursion after the residual stack.
    W.append(conv_identity(channels=16, kernel_size=3))

    residual_input_size = (1, 16, 14, 14)
    stem_input_size = (1, 1, 28, 28)
    classifier_input_size = (1, 16 * 7 * 7)

    start_time = time.time()

    m = [lipschitz_general(W[0], residual_input_size, iter=N)]

    for kk in range(len(W) - 1):
        G_list = []
        W_list = []
        for jj in range(kk + 1):
            G_list.append(G[jj])
            W_list.append(W[jj])

        Wkp1 = W[kk + 1]
        Gj = conv_identity(channels=16, kernel_size=3)
        tmp = lipschitz_res(G_list, W_list, Wkp1, Gj, residual_input_size, iter=N)

        sum_val = 0
        for jj in range(kk + 1):
            Gj = G[jj]
            G_list = []
            W_list = []
            for ii in range(jj + 1, kk + 1):
                G_list.append(G[ii])
                W_list.append(W[ii])
            Wkp1 = W[kk + 1]

            sum_val += lipschitz_res(G_list, W_list, Wkp1, Gj, residual_input_size, iter=N) * m[jj]

        m.append(tmp + 0.5 * sum_val)

    # Stem is SamePadConv2d with k=7, s=1, so padding is 3.
    # MaxPool2d has l2-Lipschitz constant <= 1.
    stem_lip = lipschitz_general(ckpt["stem.0.conv.weight"], stem_input_size, (1, 1), (3, 3), iter=N)

    # Final AvgPool2d(2,2) has l2 operator norm 1/2 for non-overlapping 2x2 average pooling.
    avgpool_lip = 0.5
    classifier_lip = lipschitz_general(ckpt["classifier.weight"], classifier_input_size, iter=N)

    elapsed_time = time.time() - start_time
    Lip = stem_lip * m[-1] * avgpool_lip * classifier_lip

    Lip_triv = stem_lip
    for ii in range(len(W) - 1):
        block_lip = 1 + lipschitz_general(G[ii], residual_input_size, iter=N) * lipschitz_general(W[ii], residual_input_size, iter=N)
        print(f"Block {ii} trivial residual factor: {block_lip}")
        Lip_triv *= block_lip
    Lip_triv *= avgpool_lip * classifier_lip

    print(f"LipLT: {Lip}")
    print(f"Trivial residual product bound: {Lip_triv}")
    print(f"Elapsed time: {elapsed_time:.3f} s")


if __name__ == "__main__":
    main()
