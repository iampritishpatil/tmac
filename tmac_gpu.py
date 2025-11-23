# tmac_gpu.py
# Milestones 1-2: GPU-batched TMAC with shared/hierarchical timescales and optional fixed taus.
# No SciPy; pure PyTorch autograd using LBFGS/Adam. Pydantic v2 for parameter handling.

from __future__ import annotations
from typing import Literal, Optional, Union, Dict, Any
from enum import Enum

import math
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator

# ---------------------------
# Fourier helpers (real-packed, orthonormal scaling)
# ---------------------------


def get_fourier_freq(
    t_max: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Returns angular frequencies (rad/sample) aligned with the real-packed basis used by real_fft/real_ifft.
    Length equals t_max; order: DC..cos(Nyq) then negative-frequency sines.
    """
    n_cos = int(np.ceil((t_max + 1) / 2))
    n_sin = int(np.floor((t_max - 1) / 2))
    w_cos = torch.arange(0, n_cos, device=device, dtype=dtype)
    w_sin = torch.arange(-n_sin, 0, device=device, dtype=dtype)
    w_vec = torch.cat([w_cos, w_sin], dim=0)
    freqs = 2.0 * math.pi / float(t_max) * w_vec
    return freqs


def real_fft(x: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
    """
    Orthonormal real-packed FFT matching the original TMAC formulation.
    x: [T] or [T,N] (real). Returns [T] or [T,N] (real) packed as cos-block then (-imag sine) block.
    """
    single_vec = x.dim() == 1
    if single_vec:
        x = x[:, None]
    if n is None:
        n = x.shape[0]
    # Complex FFT with unitary-ish normalization to match original scaling
    x_fft = torch.fft.fft(x, n=n, dim=0) / math.sqrt(n / 2.0)
    # DC
    x_fft[0, :] = x_fft[0, :] / math.sqrt(2.0)
    # Nyquist (if even)
    if n % 2 == 0:
        imx = int(math.ceil((n - 1) / 2.0))
        x_fft[imx, :] = x_fft[imx, :] / math.sqrt(2.0)
    x_hat = x_fft.real.clone()
    isin = int(math.ceil((n + 1) / 2.0))
    x_hat[isin:, :] = -x_fft[isin:, :].imag
    if single_vec:
        x_hat = x_hat[:, 0]
    return x_hat


def real_ifft(x_hat: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
    """
    Inverse of real_fft.
    x_hat: [T] or [T,N] (real-packed). Returns real time-domain [T] / [T,N].
    """
    single_vec = x_hat.dim() == 1
    if single_vec:
        x_hat = x_hat[:, None]
    if n is None:
        n = x_hat.shape[0]
    nxh = x_hat.shape[0]
    n_cos = int(math.ceil((nxh + 1) / 2.0))
    n_sin = int(math.floor((nxh - 1) / 2.0))
    x_hat = x_hat.clone()
    x_hat[0, :] = x_hat[0, :] * math.sqrt(2.0)
    if nxh % 2 == 0:
        x_hat[n_cos - 1, :] = x_hat[n_cos - 1, :] * math.sqrt(2.0)
    xfft = torch.zeros_like(x_hat, dtype=torch.complex128, device=x_hat.device)
    xfft[:] = x_hat[:]
    xfft[n_cos:, :] = torch.flip(x_hat[1 : n_sin + 1, :], dims=[0])
    xfft[1 : n_sin + 1, :] = xfft[1 : n_sin + 1, :] + 1j * torch.flip(
        x_hat[n_cos:, :], dims=[0]
    )
    xfft[n_cos:, :] = xfft[n_cos:, :] - 1j * x_hat[n_cos:, :]
    x = torch.fft.ifft(xfft, dim=0).real * math.sqrt(nxh / 2.0)
    x = x[:n, :]
    if single_vec:
        x = x[:, 0]
    return x


# ---------------------------
# Config (Pydantic v2)
# ---------------------------


class PoolingMode(str, Enum):
    fixed = "fixed"  # taus provided (scalar or per-ROI)
    global_shared = "global"
    hierarchical = "hierarchical"


class OptimizerName(str, Enum):
    lbfgs = "lbfgs"
    adam = "adam"


class DTypeName(str, Enum):
    float32 = "float32"
    float64 = "float64"


class TMACGPUOptions(BaseModel):
    # Core
    pooling: PoolingMode = Field(
        default=PoolingMode.global_shared, description="Timescale pooling mode."
    )
    tau_a: Optional[Union[float, list[float]]] = Field(
        default=None, description="Activity timescale(s): used when pooling='fixed'."
    )
    tau_m: Optional[Union[float, list[float]]] = Field(
        default=None, description="Motion timescale(s): used when pooling='fixed'."
    )
    # Hierarchical prior (log-normal via Gaussian on log τ)
    hier_mu_a: float = Field(
        default=0.0, description="Mean of log tau_a (hierarchical)."
    )
    hier_sigma_a: float = Field(
        default=0.5, gt=0, description="Std of log tau_a (hierarchical)."
    )
    hier_mu_m: float = Field(
        default=0.0, description="Mean of log tau_m (hierarchical)."
    )
    hier_sigma_m: float = Field(
        default=0.5, gt=0, description="Std of log tau_m (hierarchical)."
    )
    learn_hier_mu: bool = Field(
        default=False, description="If true, learn hierarchical means (mu_a, mu_m)."
    )
    # Optimization
    device: str = Field(
        default="mps", description="Device to use: 'cpu', 'cuda', 'mps', etc."
    )
    dtype: DTypeName = Field(default=DTypeName.float64)
    optimizer: OptimizerName = Field(default=OptimizerName.lbfgs)
    max_iter: int = Field(default=150, ge=1)
    lr: float = Field(default=0.3, gt=0)
    history_size: int = Field(default=20, ge=1)
    # Numerics
    truncate_freq: bool = Field(
        default=False,
        description="If true, drop ultrahigh frequencies using threshold rule.",
    )
    threshold: float = Field(
        default=1e8, description="Stability constant for spectral floor."
    )
    # Initialization
    init_log_tau_a: float = Field(
        default=0.0, description="Initial log tau_a (when learned)."
    )
    init_log_tau_m: float = Field(
        default=0.0, description="Initial log tau_m (when learned)."
    )

    @field_validator("tau_a", "tau_m")
    @classmethod
    def _validate_tau(cls, v):
        if v is None:
            return v
        if isinstance(v, (float, int)):
            if v <= 0:
                raise ValueError("tau must be positive")
            return float(v)
        elif isinstance(v, list):
            if any((float(x) <= 0) for x in v):
                raise ValueError("all tau entries must be positive")
            return [float(x) for x in v]
        else:
            raise ValueError("tau must be float or list of floats")


# ---------------------------
# Core batched objective and posterior
# ---------------------------


def _spectral_covariances(
    freqs: torch.Tensor,
    log_tau_a: torch.Tensor,
    log_tau_m: torch.Tensor,
    log_var_a: torch.Tensor,
    log_var_m: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute diagonal spectral covariances C_a(ω), C_m(ω) with RBF kernel.
    freqs: [F]
    log_tau_*: [N] or [1]
    log_var_*: [N]
    Returns C_a_fft, C_m_fft: [F, N]
    """
    F = freqs.shape[0]
    # Shapes
    log_tau_a = log_tau_a.view(1, -1)  # [1,N] or [1,1]
    log_tau_m = log_tau_m.view(1, -1)
    log_var_a = log_var_a.view(1, -1)
    log_var_m = log_var_m.view(1, -1)

    tau_a = torch.exp(log_tau_a)
    tau_m = torch.exp(log_tau_m)
    var_a = torch.exp(log_var_a)
    var_m = torch.exp(log_var_m)

    w2 = freqs.view(F, 1) ** 2
    # spectral densities for RBF: σ^2 τ sqrt(2π) exp(-0.5 w^2 τ^2)
    C_a = var_a * (tau_a * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * w2 * (tau_a**2))
    C_m = var_m * (tau_m * math.sqrt(2.0 * math.pi)) * torch.exp(-0.5 * w2 * (tau_m**2))

    # Numerical floor
    cutoff = 1.0 / float(threshold)
    C_a = torch.clamp(C_a, min=cutoff)
    C_m = torch.clamp(C_m, min=cutoff)
    return C_a, C_m


def _evidence_and_posterior(
    red: torch.Tensor,
    green: torch.Tensor,
    log_vars: Dict[str, torch.Tensor],
    log_tau_a: torch.Tensor,
    log_tau_m: torch.Tensor,
    truncate_freq: bool,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    """
    Vectorized evidence (sum over neurons) and posterior means.
    Inputs:
      red, green: [T,N], fold-change (baseline ~0), real.
      log_vars: dict with keys 'r_noise','g_noise','a','m' each [N] log-variance tensors.
      log_tau_a, log_tau_m: [N] or [1] log timescales.
    Returns dict with:
      'neg_evidence': scalar,
      'a_hat': [T,N] (baseline +1 to be done by caller),
      'm_hat': [T,N],
      'components': dict of per-neuron terms for diagnostics.
    """
    T, N = red.shape
    device = red.device
    dtype = red.dtype

    # FFT (real-packed)
    R_hat = real_fft(red)  # [T,N] real
    G_hat = real_fft(green)  # [T,N] real

    # Frequencies
    freqs = get_fourier_freq(T, device=device, dtype=dtype)  # [T]

    # Spectral covariances [F,N]
    C_a, C_m = _spectral_covariances(
        freqs,
        log_tau_a=log_tau_a,
        log_tau_m=log_tau_m,
        log_var_a=log_vars["a"],
        log_var_m=log_vars["m"],
        threshold=threshold,
    )

    inv_sigma_r2 = torch.exp(-log_vars["r_noise"]).view(1, N)
    inv_sigma_g2 = torch.exp(-log_vars["g_noise"]).view(1, N)

    # Precision components per frequency & neuron
    f11 = 1.0 / C_a + inv_sigma_g2
    f22 = 1.0 / C_m + inv_sigma_r2 + inv_sigma_g2
    f12 = inv_sigma_g2.expand_as(f11)

    f_det = f11 * f22 - f12**2
    # Inverse via 2x2 identity
    k = f11 - (f12**2) / f22
    f11_inv = 1.0 / k
    f22_inv = 1.0 / f22 + (f12**2) / (f22**2) / k
    f12_inv = -f12 / f22 / k

    # Log-det term (sum over freqs) + noise det
    # -( sum log f_det + sum log(C_a C_m) + T * log(sig_g^2 sig_r^2) )
    log_det_term = -(
        torch.sum(torch.log(f_det), dim=0)  # per-neuron
        + torch.sum(torch.log(C_a * C_m), dim=0)
        + T * (log_vars["g_noise"] + log_vars["r_noise"])
    )  # [N]

    # Quadratic term
    auto_corr = (
        inv_sigma_r2 * torch.sum(red**2, dim=0).view(1, N)
        + inv_sigma_g2 * torch.sum(green**2, dim=0).view(1, N)
    ).view(-1)  # [N]

    norm_R = inv_sigma_r2 * R_hat  # [F,N]
    norm_G = inv_sigma_g2 * G_hat  # [F,N]
    f_quad_mult_1 = norm_G
    f_quad_mult_2 = norm_R + norm_G

    f_quad = torch.sum(
        f11_inv * f_quad_mult_1**2
        + f22_inv * f_quad_mult_2**2
        + 2.0 * f12_inv * f_quad_mult_1 * f_quad_mult_2,
        dim=0,
    )  # [N]
    quad_term = -(auto_corr - f_quad)

    evidence_per_neuron = log_det_term + quad_term  # [N]
    evidence_sum = torch.sum(evidence_per_neuron)  # scalar

    # Posterior means in spectral domain
    A_fft = f11_inv * f_quad_mult_1 + f12_inv * f_quad_mult_2
    M_fft = f22_inv * f_quad_mult_2 + f12_inv * f_quad_mult_1

    # Back to time
    A_hat = real_ifft(A_fft)  # [T,N]
    M_hat = real_ifft(M_fft)

    return {
        "neg_evidence": -evidence_sum,  # to minimize
        "a_hat": A_hat,
        "m_hat": M_hat,
        "components": {
            "evidence_per_neuron": evidence_per_neuron.detach(),
            "log_det_term": log_det_term.detach(),
            "quad_term": quad_term.detach(),
        },
    }


# ---------------------------
# Main fit function
# ---------------------------


def fit(
    red_np: np.ndarray, green_np: np.ndarray, options: TMACGPUOptions
) -> Dict[str, Any]:
    """
    GPU-batched TMAC fit with timescale pooling (Milestones 1-2).
    - Works in fold-change units (divide by per-ROI mean, subtract 1).
    - Returns posterior means, learned parameters, and diagnostics.
    """

    if not isinstance(red_np, np.ndarray) or not isinstance(green_np, np.ndarray):
        raise TypeError("Inputs must be numpy arrays")
    if red_np.ndim == 1:
        red_np = red_np[:, None]
    if green_np.ndim == 1:
        green_np = green_np[:, None]
    if red_np.shape != green_np.shape:
        raise ValueError("red and green must have identical shape [T,N]")

    if np.any(~np.isfinite(red_np)) or np.any(~np.isfinite(green_np)):
        raise ValueError("Inputs contain NaN/Inf. Please impute/clean before fitting.")

    T, N = red_np.shape

    device = torch.device(options.device)
    dtype = torch.float64 if options.dtype == DTypeName.float64 else torch.float32

    # Fold-change normalization
    red_mean = np.mean(red_np, axis=0, keepdims=True)
    green_mean = np.mean(green_np, axis=0, keepdims=True)
    red_fc = red_np / red_mean - 1.0
    green_fc = green_np / green_mean - 1.0

    red = torch.tensor(red_fc, device=device, dtype=dtype)
    green = torch.tensor(green_fc, device=device, dtype=dtype)

    # Initialize parameters (log-variances per ROI)
    # Use sample variances in fold-change space
    var_r0 = np.var(red_fc, axis=0) + 1e-8
    var_g0 = np.var(green_fc, axis=0) + 1e-8
    var_a0 = np.maximum(var_g0, 1e-6)
    var_m0 = np.maximum(var_r0, 1e-6)

    log_var_r = torch.nn.Parameter(
        torch.tensor(np.log(var_r0), device=device, dtype=dtype)
    )
    log_var_g = torch.nn.Parameter(
        torch.tensor(np.log(var_g0), device=device, dtype=dtype)
    )
    log_var_a = torch.nn.Parameter(
        torch.tensor(np.log(var_a0), device=device, dtype=dtype)
    )
    log_var_m = torch.nn.Parameter(
        torch.tensor(np.log(var_m0), device=device, dtype=dtype)
    )

    # Timescales
    learn_log_tau_a: Optional[torch.nn.Parameter] = None
    learn_log_tau_m: Optional[torch.nn.Parameter] = None
    log_tau_a_values: torch.Tensor
    log_tau_m_values: torch.Tensor

    if options.pooling == PoolingMode.fixed:
        # tau_a / tau_m provided; accept scalar or per-ROI list
        if options.tau_a is None or options.tau_m is None:
            raise ValueError("When pooling='fixed', tau_a and tau_m must be provided.")
        if isinstance(options.tau_a, list):
            if len(options.tau_a) != N:
                raise ValueError("tau_a list must have length N.")
            tau_a_np = np.array(options.tau_a, dtype=np.float64)
        else:
            tau_a_np = np.full((N,), float(options.tau_a), dtype=np.float64)
        if isinstance(options.tau_m, list):
            if len(options.tau_m) != N:
                raise ValueError("tau_m list must have length N.")
            tau_m_np = np.array(options.tau_m, dtype=np.float64)
        else:
            tau_m_np = np.full((N,), float(options.tau_m), dtype=np.float64)

        log_tau_a_values = torch.tensor(np.log(tau_a_np), device=device, dtype=dtype)
        log_tau_m_values = torch.tensor(np.log(tau_m_np), device=device, dtype=dtype)

    elif options.pooling == PoolingMode.global_shared:
        learn_log_tau_a = torch.nn.Parameter(
            torch.tensor([options.init_log_tau_a], device=device, dtype=dtype)
        )
        learn_log_tau_m = torch.nn.Parameter(
            torch.tensor([options.init_log_tau_m], device=device, dtype=dtype)
        )
        log_tau_a_values = learn_log_tau_a
        log_tau_m_values = learn_log_tau_m

    else:  # hierarchical
        # Per-ROI log taus with Gaussian prior toward (mu, sigma). Optionally learn mu.
        learn_log_tau_a = torch.nn.Parameter(
            torch.full((N,), options.init_log_tau_a, device=device, dtype=dtype)
        )
        learn_log_tau_m = torch.nn.Parameter(
            torch.full((N,), options.init_log_tau_m, device=device, dtype=dtype)
        )
        log_tau_a_values = learn_log_tau_a
        log_tau_m_values = learn_log_tau_m

        if options.learn_hier_mu:
            hier_mu_a = torch.nn.Parameter(
                torch.tensor(options.hier_mu_a, device=device, dtype=dtype)
            )
            hier_mu_m = torch.nn.Parameter(
                torch.tensor(options.hier_mu_m, device=device, dtype=dtype)
            )
        else:
            hier_mu_a = torch.tensor(options.hier_mu_a, device=device, dtype=dtype)
            hier_mu_m = torch.tensor(options.hier_mu_m, device=device, dtype=dtype)
        hier_sigma_a = torch.tensor(options.hier_sigma_a, device=device, dtype=dtype)
        hier_sigma_m = torch.tensor(options.hier_sigma_m, device=device, dtype=dtype)

    # Collect parameters for optimizer
    params = [log_var_r, log_var_g, log_var_a, log_var_m]
    if options.pooling == PoolingMode.global_shared:
        params += [learn_log_tau_a, learn_log_tau_m]
    elif options.pooling == PoolingMode.hierarchical:
        params += [learn_log_tau_a, learn_log_tau_m]
        if isinstance(hier_mu_a, torch.nn.Parameter):
            params += [hier_mu_a]
        if isinstance(hier_mu_m, torch.nn.Parameter):
            params += [hier_mu_m]

    # Optimizer
    if options.optimizer == OptimizerName.lbfgs:
        optim = torch.optim.LBFGS(
            params,
            lr=options.lr,
            max_iter=options.max_iter,
            history_size=options.history_size,
            line_search_fn="strong_wolfe",
        )
    else:
        optim = torch.optim.Adam(params, lr=options.lr)

    loss_history = []

    def _closure():
        optim.zero_grad(set_to_none=True)

        # Hierarchical mu/sigma tensors (if not hierarchical, dummy tensors unused)
        if options.pooling == PoolingMode.hierarchical:
            # Use current values (some might be nn.Parameter)
            mu_a = (
                hier_mu_a
                if isinstance(hier_mu_a, torch.Tensor)
                else torch.tensor(options.hier_mu_a, device=device, dtype=dtype)
            )
            mu_m = (
                hier_mu_m
                if isinstance(hier_mu_m, torch.Tensor)
                else torch.tensor(options.hier_mu_m, device=device, dtype=dtype)
            )
            sigma_a = torch.tensor(options.hier_sigma_a, device=device, dtype=dtype)
            sigma_m = torch.tensor(options.hier_sigma_m, device=device, dtype=dtype)
        else:
            mu_a = mu_m = sigma_a = sigma_m = None  # type: ignore

        # Build dict of log variances
        log_vars = {
            "r_noise": log_var_r,
            "g_noise": log_var_g,
            "a": log_var_a,
            "m": log_var_m,
        }

        out = _evidence_and_posterior(
            red=red,
            green=green,
            log_vars=log_vars,
            log_tau_a=log_tau_a_values,
            log_tau_m=log_tau_m_values,
            truncate_freq=options.truncate_freq,
            threshold=options.threshold,
        )

        neg_evidence = out["neg_evidence"]

        # Hierarchical penalties (Gaussian on log tau)
        penalty = torch.tensor(0.0, device=device, dtype=dtype)
        if options.pooling == PoolingMode.hierarchical:
            penalty = (
                penalty
                + 0.5 * torch.sum(((learn_log_tau_a - mu_a) / sigma_a) ** 2)
                + 0.5 * torch.sum(((learn_log_tau_m - mu_m) / sigma_m) ** 2)
            )

        loss = neg_evidence + penalty
        loss.backward()
        loss_history.append(loss.detach().item())
        return loss

    if options.optimizer == OptimizerName.lbfgs:
        optim.step(_closure)
    else:
        for _ in range(options.max_iter):
            _closure()
            optim.step()

    # Final posterior with learned parameters
    final_out = _evidence_and_posterior(
        red=red,
        green=green,
        log_vars={
            "r_noise": log_var_r,
            "g_noise": log_var_g,
            "a": log_var_a,
            "m": log_var_m,
        },
        log_tau_a=log_tau_a_values,
        log_tau_m=log_tau_m_values,
        truncate_freq=options.truncate_freq,
        threshold=options.threshold,
    )
    a_hat_fc = final_out["a_hat"] + 1.0  # return baseline ~1
    m_hat = final_out["m_hat"]

    # Gather learned params
    result: Dict[str, Any] = {
        "a_hat": a_hat_fc.detach().cpu().numpy(),
        "m_hat": m_hat.detach().cpu().numpy(),
        "loss_history": loss_history,
        "log_vars": {
            "r_noise": log_var_r.detach().cpu().numpy(),
            "g_noise": log_var_g.detach().cpu().numpy(),
            "a": log_var_a.detach().cpu().numpy(),
            "m": log_var_m.detach().cpu().numpy(),
        },
        "taus": {},
    }

    if options.pooling == PoolingMode.fixed:
        result["taus"]["tau_a"] = torch.exp(log_tau_a_values).detach().cpu().numpy()
        result["taus"]["tau_m"] = torch.exp(log_tau_m_values).detach().cpu().numpy()
    elif options.pooling == PoolingMode.global_shared:
        result["taus"]["tau_a"] = float(torch.exp(log_tau_a_values).item())
        result["taus"]["tau_m"] = float(torch.exp(log_tau_m_values).item())
    else:
        result["taus"]["tau_a"] = torch.exp(learn_log_tau_a.detach()).cpu().numpy()
        result["taus"]["tau_m"] = torch.exp(learn_log_tau_m.detach()).cpu().numpy()
        if isinstance(hier_mu_a, torch.Tensor):
            result["taus"]["hier_mu_a"] = float((hier_mu_a.detach().cpu().item()))
        else:
            result["taus"]["hier_mu_a"] = float(options.hier_mu_a)
        if isinstance(hier_mu_m, torch.Tensor):
            result["taus"]["hier_mu_m"] = float((hier_mu_m.detach().cpu().item()))
        else:
            result["taus"]["hier_mu_m"] = float(options.hier_mu_m)
        result["taus"]["hier_sigma_a"] = float(options.hier_sigma_a)
        result["taus"]["hier_sigma_m"] = float(options.hier_sigma_m)

    # Add per-neuron evidence for diagnostics
    result["evidence_per_neuron"] = (
        final_out["components"]["evidence_per_neuron"].cpu().numpy()
    )
    result["log_det_term"] = final_out["components"]["log_det_term"].cpu().numpy()
    result["quad_term"] = final_out["components"]["quad_term"].cpu().numpy()

    return result
