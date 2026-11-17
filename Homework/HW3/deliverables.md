# HW3 – Reachability Analysis of Neural Feedback Loops

## The big idea

HW2 verified a network by itself. HW3 puts the network *in a control loop*: a neural controller picks an action, the system's dynamics move the state, and the loop repeats. The question changes from "what can this network output?" to "**where can this robot go over time?**" — and, for safety, "**can it ever hit the obstacle?**"

Two ways to answer it:
- **Forward reachability**: start from the set of initial states and bound where the system can be at each future step.
- **Backward reachability**: start from the set we must avoid and bound which states lead *into* it. If the robot doesn't start in one of those sets, we've proved it won't collide on the next step.

The system is a 2D ground robot (`GroundRobot`, x_{t+1} = A x_t + B u_t) with a trained obstacle-avoidance controller, from the `nfl_veripy` library.

## Key concepts

- **Closed-loop step**: `step_1(x) = dynamics(x, controller(x))`. We bound this whole function with the HW2 tools (IBP, CROWN via `jax_verify`).
- **Iterative vs. one-shot**: iterative feeds each step's bounding box into the next step, so the error piles up (the wrapping effect). One-shot bounds all T steps at once from the initial set, which should be tighter.
- **Bifurcating dynamics**: when the initial set spans y = 0, some trajectories go above the obstacle and some go below. A single box can't capture two branches, so forward bounds blow up.
- **Backreachable set (BR)**: states from which *some* allowed control reaches the target. Found with LPs over the control limits; ignores the controller.
- **Backprojection set (BP)**: states from which *our* controller reaches the target. Tighter than BR; found by adding CROWN's linear bounds on the controller as extra LP constraints.
- **Branch and bound (partitioning)**: split the BR set into smaller cells and redo BP on each. CROWN's relaxation is tighter on small regions, so the union of the pieces is a much better bound.

## What the assignment asks for

| Part | Task | Deliverables |
|---|---|---|
| 1a | 1-step forward reachable set | `compute_1step_reachable_set`; bounds plotted over Monte Carlo samples |
| 1b | T-step, iterative | `compute_Tstep_reachable_set_iterative` |
| 1c | T-step, one-shot | `compute_Tstep_reachable_set_one_shot`; comment on bounds and runtime vs. iterative |
| 1d | Bifurcating initial set | Sample plot over 10 steps; one-shot bounds over 2 steps |
| 2a | "True" backprojection set by sampling | `get_true_backprojection_set`; plot with the target set |
| 2b | Backreachable set via LPs | `get_backreachable_set`; plot BR, target, and sampled BP |
| 2c | Backprojection set via CROWN + LP | `get_backprojection_set`; plot BP, BR, and sampled BP |
| 2d | Uniform partitioning | Results for several partition counts |
| 3 (extra credit) | Any extension (open-ended) | e.g., the other BP version, B&B for forward sets, JIT, a new policy, RL rewards, noisy observations |

## What our submitted notebook shows

Submitted version: `Cindy_Fanghsin_Li_EECE_CS_7268_HW3.ipynb`.

- **1b–1c**: we ran iterative and one-shot with three bound methods from `jax_verify` (CROWN+IBP, IBP, forward CROWN) over 10 steps and timed them:

| Method | Iterative | One-shot | One-shot / iterative volume, step 10 |
|---|---|---|---|
| CROWN+IBP | 2.49 s | 4.14 s | ~35,000× |
| IBP | 0.92 s | 0.34 s | 1.0× |
| Forward CROWN | 1.80 s | 1.31 s | ~1.9 million× |

- **1d**: on the bifurcating case, CROWN+IBP and IBP bounds get loose by step 2; forward CROWN holds until step 4–5.
- **2a**: 227 of 100,000 samples land in the target set in one step.
- **2b–2d**: BR via LPs, BP via CROWN linear bounds on the controller, then uniform 2×2 partitioning of the BR set.
- **3**: we asked for extra credit for comparing the three `jax_verify` methods.

Worth a second look:
- The volume ratios say our one-shot sets are much *looser* than iterative for both CROWN methods. The assignment expects the opposite, so the one-shot implementation (or the ratio) may have a bug.
- The written comments don't match the printed numbers: for CROWN+IBP the text says iterative is slower, but the timings show one-shot is slower; and the 1c conclusion calls CROWN+IBP the fastest, but IBP is the fastest in both columns.
- 2d shows only one partition count ([2, 2]); the assignment asks for several.
