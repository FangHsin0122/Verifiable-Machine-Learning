# Models

Trained networks used in the homework.

| File | What it is |
|---|---|
| [mnist1.nnet](mnist1.nnet) | MNIST classifier, 784 → 25 → 10, in `.nnet` format (Kyle Julian's NNet format). Used in HW2 |
| [cartpole_nnet.nnet](cartpole_nnet.nnet) | Cartpole controller, 4 → 16 → 16 → 16 → 2, in `.nnet` format. Used in HW2 and as the default input of [lp_verification.py](../Homework/HW2/lp_verification.py) |
| [lenet_mnist_model.pth](lenet_mnist_model.pth) | LeNet MNIST weights (PyTorch `state_dict`) |
| [lenet_mnist_model (1).pth](<lenet_mnist_model (1).pth>) | Identical copy of the file above |

The HW2 code opens the `.nnet` files by bare filename. See [../Homework/HW2/README.md](../Homework/HW2/README.md) for how to run it from here.
