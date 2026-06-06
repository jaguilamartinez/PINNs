# Physics-Informed Neural Networks

Three notebooks implementing PINNs for classic 1D PDEs using PyTorch. The idea is simple: instead of training on data, you train a neural network to satisfy a differential equation by penalizing violations of the PDE, boundary conditions, and initial conditions directly in the loss function.

## Notebooks

| Notebook | Equation | Exact solution |
|---|---|---|
| `burguers_equation.ipynb` | $u_t + u\,u_x = \nu\,u_{xx}$ | Via Cole–Hopf transform |
| `wave_equation.ipynb` | $u_{tt} = c^2 u_{xx}$ | $\sin(\pi x)\cos(\pi ct)$ |
| `heat_equation.ipynb` | $u_t = \alpha\,u_{xx}$ | $\sin(\pi x)\,e^{-\alpha\pi^2 t}$ |

**Burgers'** is the hardest — nonlinear, forms shocks, no closed-form you can just evaluate. **Wave** introduces a second-order time derivative, which means two initial conditions and an extra loss term for the velocity IC. **Heat** is the cleanest benchmark: linear, first-order in time, and the PINN converges fast.

## How it works

Each PINN is a small fully-connected network with Tanh activations (smooth enough for second-order autograd). Training happens in two phases:

1. **Adam** — fast pre-conditioning to get into a reasonable basin
2. **L-BFGS** — quasi-Newton fine-tuning that drives the residual down to near machine precision

The PDE residual is computed at a grid of collocation points using `torch.autograd.grad`. No finite differences, no discretization — derivatives are exact.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib tqdm jupyter
```

Then open any notebook with `jupyter notebook` and run all cells.

## Results

The wave and heat notebooks compare the PINN prediction against the known exact solution and plot the absolute error. For Burgers', the network captures the shock formation at $\nu = 0.01/\pi$ without any special treatment of the discontinuity.
