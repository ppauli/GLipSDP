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
    parser = argparse.ArgumentParser(description="LipLT bound for the FC-R18 model used in MODEL_FACTORIES.")
    parser.add_argument("--weights", type=str, default="FC-R18_instance1.pt")
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--iter", type=int, default=500)
    args = parser.parse_args()

    ckpt = load_state_dict(args.weights)
    N = args.iter
    num_blocks = args.num_blocks

    # Model:
    # Flatten -> input_layer -> ReLU
    # blocks.i: fc1 -> ReLU -> fc2 -> + identity -> ReLU
    # output_layer
    W = [ckpt["input_layer.weight"]]
    G = [torch.eye(64)]
    H = [torch.zeros(64, 28 * 28)]

    for i in range(num_blocks):
        W.append(ckpt[f"blocks.{i}.fc1.weight"])
        G.append(ckpt[f"blocks.{i}.fc2.weight"])
        H.append(torch.eye(64))

    W.append(ckpt["output_layer.weight"])

    input_size_vec = [(1, 28 * 28)] + [(1, 64) for _ in range(num_blocks + 1)]

    start_time = time.time()

    Hhat = []
    for ii in range(len(W) - 1):
        Hhat.append(H[ii] + 0.5 * torch.matmul(G[ii], W[ii]))

    m = [lipschitz_general(W[0], input_size_vec[0], iter=N)]

    for kk in range(len(W) - 1):
        H_list = []
        for jj in range(kk + 1):
            H_list.append(Hhat[jj])
        H_list.append(W[kk + 1])

        tmp = lipschitz_general(H_list, input_size_vec[0], iter=N)

        sum_val = 0
        for jj in range(kk + 1):
            H2_list = [G[jj]]
            input_size = input_size_vec[jj + 1]
            for ii in range(jj + 1, kk + 1):
                H2_list.append(Hhat[ii])
            H2_list.append(W[kk + 1])

            sum_val += lipschitz_general(H2_list, input_size, iter=N) * m[jj]

        m.append(tmp + 0.5 * sum_val)

    elapsed_time = time.time() - start_time
    Lip = m[-1]

    Lip_triv = lipschitz_general(W[0], input_size_vec[0], iter=N)
    for ii in range(1, len(W) - 1):
        Lip_triv *= 1 + lipschitz_general(G[ii], input_size_vec[ii], iter=N) * lipschitz_general(W[ii], input_size_vec[ii], iter=N)
    Lip_triv *= lipschitz_general(W[-1], input_size_vec[-1], iter=N)

    print(f"LipLT: {Lip}")
    print(f"Trivial residual product bound: {Lip_triv}")
    print(f"Elapsed time: {elapsed_time:.3f} s")


if __name__ == "__main__":
    main()
