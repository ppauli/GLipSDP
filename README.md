# GLipSDP
This repository contains the code used for the paper "Lipschitz constant estimation for general convolutional neural network architectures using control tools" (preprint: https://arxiv.org/abs/2405.01125). The code is written in Matlab and uses Yalmip (https://yalmip.github.io/) and the solver Mosek (https://www.mosek.com/).


1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Note: `requirements.txt` lists core libraries but not a specific PyTorch build. If you have CUDA and want a CUDA-enabled PyTorch, install PyTorch following the official instructions at https://pytorch.org/get-started/locally/ before running the training.

3. Run the training script:

```bash
python train_networks.py
```

If you saw the error `ModuleNotFoundError: No module named 'sklearn'` previously, installing scikit-learn into your active environment fixes it:

```bash
pip install scikit-learn
```

Troubleshooting:
- Confirm the interpreter: `which python` should point to `.venv/bin/python` when activated.
- Check installed packages: `python -m pip list`.
- For GPU support, pick the correct CUDA-enabled PyTorch install command from the PyTorch site.
