# example_gpu.py
# Example usage for TMAC-GPU Milestones 1-2 on synthetic data.
# Mirrors the original example but runs the batched GPU fitter and shows
# (1) global-shared taus (learned) and (2) fixed taus provided by the user.

import numpy as np
import matplotlib.pyplot as plt
import torch

# Import your original synthetic generator and preprocessing
# (assumes the original 'tmac' package from your repo is importable)
import tmac.preprocessing as tp
from tmac.synthetic_data import generate_synthetic_data, col_corr, ratio_model

from tmac_gpu_with_exp_kern import (
    TMACGPUOptions,
    fit,
    PoolingMode,
    DTypeName,
    OptimizerName,
)

# --- Synthetic data ---
num_ind = 1000
num_neurons = 100
mean_r = 30
mean_g = 30
variance_noise_r_true = 0.2**2
variance_noise_g_true = 0.2**2
variance_a_true = 0.3**2
variance_m_true = 0.3**2
tau_a_true = 4.0
tau_m_true = 1.0
frac_nan = 0.0
beta = 20

red_bleached, green_bleached, a_true, m_true = generate_synthetic_data(
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

# --- Preprocess (same as original example) ---
red = tp.photobleach_correction(red_bleached)
green = tp.photobleach_correction(green_bleached)
red = tp.interpolate_over_nans(red)[0]
green = tp.interpolate_over_nans(green)[0]

# --- Run TMAC-GPU: (A) global-shared taus (learned) ---
opts_global = TMACGPUOptions(
    dt=1 / 30.0,
    pooling=PoolingMode.global_shared,
    device="mps" if torch.cuda.is_available() else "cpu",
    dtype=DTypeName.float64,
    optimizer=OptimizerName.lbfgs,
    max_iter=1000,
    init_log_tau_a=np.log(tau_a_true),  # good init
    init_log_tau_m=np.log(tau_m_true),
)
res_global = fit(red, green, options=opts_global)
a_hat_global = res_global["a_hat"]
m_hat_global = res_global["m_hat"]
print("Global taus learned:", res_global["taus"])

# --- Run TMAC-GPU: (B) fixed taus provided by user ---
opts_fixed = TMACGPUOptions(
    dt=1 / 30.0,
    pooling=PoolingMode.fixed,
    tau_a=tau_a_true,
    tau_m=tau_m_true,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=DTypeName.float64,
    optimizer=OptimizerName.lbfgs,
    max_iter=10,  # no learning of taus; still learns variances in 1-2 steps with LBFGS closure
)
res_fixed = fit(red, green, options=opts_fixed)
a_hat_fixed = res_fixed["a_hat"]
m_hat_fixed = res_fixed["m_hat"]
print("Fixed taus used:", res_fixed["taus"])

# --- Baseline ratio for comparison ---
ratio = ratio_model(red, green, tau_a_true / 2)

# --- Metrics ---
ratio_r2 = col_corr(a_true, ratio) ** 2
tmac_r2_global = col_corr(a_true, a_hat_global) ** 2
tmac_r2_fixed = col_corr(a_true, a_hat_fixed) ** 2

print("Median r^2 (ratio):  ", np.median(ratio_r2))
print("Median r^2 (global): ", np.median(tmac_r2_global))
print("Median r^2 (fixed):  ", np.median(tmac_r2_fixed))

# --- Plots (single neuron) ---
plot_ind = 1
plot_start = 15
plot_time = 1000

plt.figure(figsize=(9, 7))
ax = plt.subplot(3, 1, 1)
green_fc = green / np.mean(green, axis=0)
red_fc = red / np.mean(red, axis=0)
plt.plot(green_fc[plot_start : plot_start + plot_time, plot_ind])
plt.plot(red_fc[plot_start : plot_start + plot_time, plot_ind])
plt.axhline(1, color="k", linewidth=0.5)
plt.legend(["green", "red"])
ax.set_title("Fold-change signals")

ax = plt.subplot(3, 1, 2)
plt.plot(a_true[plot_start : plot_start + plot_time, plot_ind], label="a_true")
plt.plot(
    a_hat_global[plot_start : plot_start + plot_time, plot_ind], label="a_hat (global)"
)
plt.plot(
    a_hat_fixed[plot_start : plot_start + plot_time, plot_ind],
    label="a_hat (fixed)",
    alpha=0.7,
)
plt.axhline(1, color="k", linewidth=0.5)
plt.legend()
ax.set_title("Activity")

ax = plt.subplot(3, 1, 3)
plt.plot(m_true[plot_start : plot_start + plot_time, plot_ind], label="m_true")
plt.plot(
    m_hat_global[plot_start : plot_start + plot_time, plot_ind], label="m_hat (global)"
)
plt.plot(
    m_hat_fixed[plot_start : plot_start + plot_time, plot_ind],
    label="m_hat (fixed)",
    alpha=0.7,
)
plt.axhline(0, color="k", linewidth=0.5)
plt.legend()
ax.set_title("Motion")
plt.tight_layout()
plt.show()

# --- Violin plots of r^2 improvements ---
plt.figure(figsize=(9, 4))
ax = plt.subplot(1, 2, 1)
plt.violinplot([ratio_r2, tmac_r2_global, tmac_r2_fixed])
ax.set_ylim([0, 1])
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["ratio", "global", "fixed"])
plt.ylabel("correlation squared")
plt.title("Per-ROI r^2 vs a_true")

ax = plt.subplot(1, 2, 2)
plt.violinplot([tmac_r2_global - ratio_r2, tmac_r2_fixed - ratio_r2])
lims = np.array(ax.get_ylim())
lim_to_use = np.max(np.abs(lims))
ax.set_ylim([-lim_to_use, lim_to_use])
plt.plot([0.5, 2.5], [0, 0], "-k")
ax.set_xticks([1, 2])
ax.set_xticklabels(["global - ratio", "fixed - ratio"])
plt.tight_layout()
plt.show()
