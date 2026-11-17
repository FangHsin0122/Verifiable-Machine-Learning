# HW2 – Formal Verification

## The big idea

HW1 showed that testing can't prove a network is safe. HW2 is the fix: given a *set* of inputs (a box around a nominal input), compute guaranteed bounds on *every* output the network can produce. If those bounds stay inside the safe region (for example, the predicted class never changes), we have a proof, not just evidence.

The core trade-off of the whole field shows up here: **tight bounds cost more compute, cheap bounds are loose.** We implement the methods ourselves first (sampling, IBP, LP), then use state-of-the-art libraries (`jax_verify`, `auto_LiRPA`) that do the same thing faster and tighter with CROWN.

## Key concepts

- **Under-approximation vs. over-approximation**: sampling only shows outputs we happened to hit (an *inner* estimate, so it can't prove anything). Verification methods give *outer* bounds that are guaranteed to contain every possible output.
- **Interval Bound Propagation (IBP)**: push an input box through each layer with interval arithmetic. Very fast, but the bounds grow quickly with depth.
- **LP relaxation of ReLU**: use intermediate bounds to sort each neuron into active, inactive, or uncertain; replace uncertain ReLUs with the triangle relaxation; solve one LP per output min/max.
- **Robustness verification**: prove the classifier picks the same class for every input in the box. Answers are `Robust` (proof found) or `Unsure` (no proof, which does *not* mean the model isn't robust).
- **CROWN (backward linear bound propagation)**: bound each layer with linear functions of the input and propagate them backward. Usually much tighter than IBP and much cheaper than LP.
- **NNet format**: the framework-agnostic text format the networks (`cartpole_nnet.nnet`, `mnist1.nnet`) are stored in.

## What the assignment asks for

| Problem | Task | Deliverables |
|---|---|---|
| 1 | Sampling-based output range | `output_range_sampling`; cartpole output samples + bounding box for several `num_samples` |
| 2 | IBP | `verify_ibp`; printed output range; IBP box plotted over the samples (to show how loose it is) |
| 3 | LP-based verification (cvxpy) | `verify_lp`: one LP per side of the output box |
| 4 | Classifier robustness | `verify_classification_robustness_lp`; largest ε we can verify, for cartpole (treated as a classifier) and MNIST |
| 5 | `jax_verify` | Output bounds with backward CROWN |
| 6 | `auto_LiRPA` | Output bounds with backward CROWN |

## What our final notebook shows

Final version: `Fanghsin_Li__eece7268_hw2(OK).ipynb` (run in Colab). Cartpole network: 4 → 16 → 16 → 16 → 2, input box ε = 0.1.

| Method | Output 1 range | Output 2 range | Kind |
|---|---|---|---|
| Sampling (100k samples) | [11.21, 16.20] | [−4.16, −1.09] | inner estimate |
| CROWN (`jax_verify`) | [10.74, 17.72] | [−4.76, −0.23] | outer, guaranteed |
| LP (ours) | [0.50, 25.71] | [−11.84, 5.99] | outer, guaranteed |
| IBP (ours, matches `jax_verify` IBP) | [−25.61, 52.52] | [−32.46, 23.75] | outer, guaranteed |

- IBP is the loosest by far, and our IBP matches `jax_verify`'s exactly.
- **Robustness (P4)**: cartpole verifies up to ε = 0.08 (`Unsure` from 0.09). MNIST verifies up to ε = 2.8 (`Unsure` from 2.85), on an input box of [−2, 2] per pixel.
- **P6**: the `auto_LiRPA` CROWN cell is written but has no saved output.

Worth a second look:
- Our LP bounds came out *looser* than CROWN. With the same intermediate bounds, the triangle-relaxation LP should be at least as tight as CROWN, so our intermediate bounds (probably from IBP) may be what's making the LP loose.
- The MNIST ε values (up to 2.8) are very large for pixel inputs. It's worth checking what scale the MNIST inputs are in and how `input_range_mnist` was built.
