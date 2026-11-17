# HW2 – Formal Verification

| File | What it is |
|---|---|
| [EECE7268_HW2.ipynb](EECE7268_HW2.ipynb) | Blank assignment |
| [Fanghsin_Li_EECE7268_HW2.ipynb](Fanghsin_Li_EECE7268_HW2.ipynb) | Identical copy of the blank assignment |
| [Fanghsin_Li__eece7268_hw2.ipynb](Fanghsin_Li__eece7268_hw2.ipynb) | Our completed HW2 (Oct 6), run in Google Colab |
| [Copy of (MINE)Fanghsin_Li_EECE7268_HW2.ipynb](<Copy of (MINE)Fanghsin_Li_EECE7268_HW2.ipynb>) | Working copy from Drive (Oct 7, 01:19), 60 cells |
| [(MINE)Fanghsin_Li_EECE7268_HW2.ipynb](<(MINE)Fanghsin_Li_EECE7268_HW2.ipynb>) | Later working copy from Drive (Oct 7, 02:27), 71 cells; a different layout from the completed notebook above |
| [Fanghsin_Li__eece7268_hw2(OK).ipynb](<Fanghsin_Li__eece7268_hw2(OK).ipynb>) | Drive version of our completed HW2, last edited Nov 2. Same as the Oct 6 notebook except it pins `jax==0.4.28` and has small edits to the cartpole and jax forward-pass cells |
| [lp_verification.py](lp_verification.py) | LP-based verification: loads a ReLU network from a `.nnet` file and uses cvxpy to find the min/max of each output over an input box |

## Input files

The networks these files use are in [../../Models/](../../Models/README.md):
- `mnist1.nnet` and `cartpole_nnet.nnet`.
- The blank template and `lp_verification.py` open them by bare filename, so they expect the `.nnet` file in the working directory. To run `lp_verification.py` as is, we run it from `Models/` (`python "../Homework/HW2/lp_verification.py"`) or copy the `.nnet` file next to it.
- Our completed notebook loads them from `/content/drive/MyDrive/` (Colab), so it doesn't depend on where they sit here.
