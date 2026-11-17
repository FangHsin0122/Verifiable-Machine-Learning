# HW1 – Adversarial Examples & Decision Boundaries

## The big idea

A neural network can look accurate on a test set and still be easy to fool. HW1 shows this from the attacker's side: we train small networks, then break them with tiny input changes that a human wouldn't notice. This is the motivation for the rest of the course. If testing on samples can't tell us how a network behaves on *every* nearby input, we need verification (HW2) and reachability analysis (HW3).

## Key concepts

- **Adversarial example**: an input changed by a small perturbation (bounded in some norm, here mostly ℓ∞) that flips the model's prediction.
- **Model-agnostic vs. gradient-based attacks**: constant offsets and random noise don't look at the model; FGSM uses the model's gradient, so it finds the weak direction and does much more damage for the same perturbation size.
- **FGSM / Targeted FGSM**: one step of size ε in the direction of the sign of the loss gradient. The targeted version steps *toward* a chosen wrong class instead of just away from the right one.
- **Learned vs. true decision boundary**: when we know the true rule (a circle of radius 3), we can see exactly where the network's boundary is wrong, and how architecture, activation, regularization, and data sampling change it.

## What the assignment asks for

| Part | Task | Deliverables |
|---|---|---|
| 1a | Train an MNIST classifier (Flatten → 128 ReLU → 10) | Train/val loss curves; correct and incorrect test examples |
| 1b | Constant-offset attack (`c * ones(28, 28)`, clipped to [0, 1]) | Accuracy vs. magnitude; examples that fool the model but look fine to us |
| 1c | Random-noise attack | Accuracy vs. magnitude; constant vs. noise comparison |
| 1d | FGSM, written from scratch | Accuracy vs. ε; comparison with 1b/1c |
| 1e | Targeted FGSM, written from scratch | Accuracy vs. ε per target class; successful targeted examples; which targets are easier |
| 2a | Generate 2D data labeled by ‖x‖₂ ≤ 3 | Plot of training set and true boundary |
| 2b | Train a binary classifier | Train/val loss curves |
| 2c | Compare boundaries | True vs. learned boundary plot; how model/training choices change it |
| 3 (extra credit) | White-box ℓ∞ attack on a pre-trained CIFAR-10 ResNet20 | `white_box_attack` function, ranked on a Gradescope leaderboard (< 40% accuracy earns credit) |

## What our final notebook covers

Final version: `doneEECE 7268_hw1_Fanghsin Li.ipynb`.

- **1a**: 784 → 128 → 10 network, 10 epochs; val loss drops from 0.20 to about 0.085.
- **1b–1c**: our takeaway: a constant offset shifts every pixel together (like changing brightness), so accuracy drops faster than with scattered random noise.
- **1d**: FGSM is far stronger. At ε = 0.1 it pushes accuracy down to about 20%, while constant and noise attacks barely change it.
- **2a–2c**: 2 → 64 → 32 → 1 network; plots of the true vs. learned boundary.

Gaps in our version:
- **1e (Targeted FGSM)** has only the heading. No code or answers in any of our HW1 notebooks.
- **2c** has the boundary plot but no written discussion of how architecture/training changes the boundary.
- **Problem 3 (extra credit)** wasn't attempted.
