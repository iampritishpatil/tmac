import numpy as np
import torch
from tmac_gpu_old import TMACGPU, TMACOptions

# Fake small demo data (replace with your bleach-corrected arrays)
T, N = 1024, 16
# rng = np.random.default_rng(0)
# red = 1000 + 20*rng.standard_normal((T,N))
# green = 1200 + 40*rng.standard_normal((T,N))


from matplotlib import pyplot as plt
import tmac.models as tm
import tmac.preprocessing as tp
from tmac.synthetic_data import generate_synthetic_data, col_corr, ratio_model


# set the parameters of the synthetic data
num_ind = T
num_neurons = N
mean_r = 20
mean_g = 30
variance_noise_r_true = 0.2**2
variance_noise_g_true = 0.2**2
variance_a_true = 0.3**2
variance_m_true = 0.3**2
tau_a_true = 3
tau_m_true = 3
frac_nan = 0.05
beta = 20

# generate synthetic data
red, green, a_true, m_true = generate_synthetic_data(
    num_ind,
    num_neurons,
    mean_r,
    mean_g,
    variance_noise_r_true,
    variance_noise_g_true,
    variance_a_true,
    variance_m_true,
    tau_a_true,
    tau_m_true,
    frac_nan=frac_nan,
    beta=beta,
    multiplicative=False,
)


opts = TMACOptions(
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device="mps" if torch.mps.is_available() else "cpu",
    dtype=torch.float32,
    optimizer="lbfgs",  # or "adam"
    max_iters=20,
    pooling="global",
    gcamp_tau_decay=None,  # or a float like 1.5
    init_bleed=0.001,
    z_model="rank1",  # or None
    vst="none",
)

model = TMACGPU(opts)
out = model.fit(red, green)
print("Keys:", out.keys())
print("a_hat shape:", out["a_hat"].shape)
print("params:", out["params"])
