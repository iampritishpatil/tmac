import numpy as np
import torch
import time
from scipy import optimize
from scipy.stats import norm
import tmac.probability_distributions as tpd
import tmac.fourier as tfo
import tmac.optimization as opt
import tmac.preprocessing as pp


def tmac_ac(red_np, green_np, optimizer="BFGS", verbose=False, truncate_freq=True):
    """Implementation of the Two-channel motion artifact correction method (TMAC)

    This is tmac_ac because it is the additive and circular boundary version
    This code takes in imaging fluoresence data from two simultaneously recorded channels and attempts to remove
    shared motion artifacts between the two channels

    Args:
        red_np: numpy array, [time, neurons], activity independent channel
        green_np: numpy array, [time, neurons], activity dependent channel
        optimizer: string, scipy optimizer
        verbose: boolean, if true, outputs when inference is complete on each neuron and estimates time to finish
        truncate_freq: boolean, if true truncates low amplitude frequencies in Fourier domain. This should give the same
            results but may give sensitivity to the initial conditions

    Returns: a dictionary containing: all the inferred parameters of the model
    """

    # optimization is performed using Scipy optimize, so all tensors should stay on the CPU
    device = "cpu"
    dtype = torch.float64

    red_np = pp.check_input_format(red_np)
    green_np = pp.check_input_format(green_np)

    red_nan = np.any(np.isnan(red_np))
    red_inf = np.any(np.isinf(red_np))
    green_nan = np.any(np.isnan(green_np))
    green_inf = np.any(np.isinf(green_np))

    if red_nan or red_inf or green_nan or green_inf:
        raise Exception("Input data cannot have any nan or inf")

    if red_np.shape != green_np.shape:
        raise Exception("red and green matricies must be the same shape")

    # convert data to units of fold mean and subtract mean
    mean_red = np.mean(red_np, axis=0)
    mean_green = np.mean(green_np, axis=0)
    red_np = red_np / mean_red - 1
    green_np = green_np / mean_green - 1

    # convert to tensors and fourier transform
    red = torch.tensor(red_np, device=device, dtype=dtype)
    green = torch.tensor(green_np, device=device, dtype=dtype)
    red_fft = tfo.real_fft(red)
    green_fft = tfo.real_fft(green)

    # estimate all model parameters from the data
    variance_r_noise_init = np.var(red_np, axis=0)
    variance_g_noise_init = np.var(green_np, axis=0)
    variance_a_init = np.var(green_np, axis=0)
    variance_m_init = np.var(red_np, axis=0)

    # initialize length scale
    length_scale_a_init = np.ones(red_np.shape[1])
    length_scale_m_init = np.ones(red_np.shape[1])

    # preallocate space for all the training variables
    a_trained = np.zeros(red_np.shape)
    m_trained = np.zeros(red_np.shape)
    variance_r_noise_trained = np.zeros(variance_r_noise_init.shape)
    variance_g_noise_trained = np.zeros(variance_g_noise_init.shape)
    variance_a_trained = np.zeros(variance_a_init.shape)
    length_scale_a_trained = np.zeros(length_scale_a_init.shape)
    variance_m_trained = np.zeros(variance_m_init.shape)
    length_scale_m_trained = np.zeros(length_scale_m_init.shape)

    # loop through each neuron and perform inference
    start = time.time()
    for n in range(red_np.shape[1]):
        # get the initial values for the hyperparameters of this neuron
        # All hyperparameters are positive, so we fit them in log space
        evidence_training_variables = np.log(
            [
                variance_r_noise_init[n],
                variance_g_noise_init[n],
                variance_a_init[n],
                length_scale_a_init[n],
                variance_m_init[n],
                length_scale_m_init[n],
            ]
        )

        # define the evidence loss function. This function takes in and returns pytorch tensors
        def evidence_loss_fn(training_variables):
            return -tpd.tmac_evidence_and_posterior(
                red[:, n],
                red_fft[:, n],
                training_variables[0],
                green[:, n],
                green_fft[:, n],
                training_variables[1],
                training_variables[2],
                training_variables[3],
                training_variables[4],
                training_variables[5],
                truncate_freq=truncate_freq,
            )

        trained_variances = opt.scipy_minimize_with_grad(
            evidence_loss_fn,
            evidence_training_variables,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
        )

        # calculate the posterior values
        # The posterior is gaussian so we don't need to optimize, we find a and m in one step
        trained_variance_torch = torch.tensor(
            trained_variances.x, dtype=dtype, device=device
        )
        a, m = tpd.tmac_evidence_and_posterior(
            red[:, n],
            red_fft[:, n],
            trained_variance_torch[0],
            green[:, n],
            green_fft[:, n],
            trained_variance_torch[1],
            trained_variance_torch[2],
            trained_variance_torch[3],
            trained_variance_torch[4],
            trained_variance_torch[5],
            calculate_posterior=True,
            truncate_freq=truncate_freq,
        )

        a_trained[:, n] = a.numpy()
        m_trained[:, n] = m.numpy()
        variance_r_noise_trained[n] = torch.exp(trained_variance_torch[0]).numpy()
        variance_g_noise_trained[n] = torch.exp(trained_variance_torch[1]).numpy()
        variance_a_trained[n] = torch.exp(trained_variance_torch[2]).numpy()
        length_scale_a_trained[n] = torch.exp(trained_variance_torch[3]).numpy()
        variance_m_trained[n] = torch.exp(trained_variance_torch[4]).numpy()
        length_scale_m_trained[n] = torch.exp(trained_variance_torch[5]).numpy()

        if verbose:
            decimals = 1e3
            # print out timing
            elapsed = time.time() - start
            remaining = elapsed / (n + 1) * (red_np.shape[1] - (n + 1))
            elapsed_truncated = np.round(elapsed * decimals) / decimals
            remaining_truncated = np.round(remaining * decimals) / decimals
            print(str(n + 1) + "/" + str(red_np.shape[1]) + " neurons complete")
            print(
                str(elapsed_truncated)
                + "s elapsed, estimated "
                + str(remaining_truncated)
                + "s remaining"
            )

    trained_variables = {
        "a": a_trained,
        "m": m_trained,
        "variance_r_noise": variance_r_noise_trained,
        "variance_g_noise": variance_g_noise_trained,
        "variance_a": variance_a_trained,
        "length_scale_a": length_scale_a_trained,
        "variance_m": variance_m_trained,
        "length_scale_m": length_scale_m_trained,
    }

    return trained_variables


def initialize_length_scale(y):
    """Function to fit a Gaussian to the autocorrelation of y

    Args:
        y: numpy vector

    Returns: Standard deviation of a Gaussian fit to the autocorrelation of y
    """

    x = np.arange(-len(y) / 2, len(y) / 2) + 0.5
    y_z_score = (y - np.mean(y)) / np.std(y)
    y_corr = np.correlate(y_z_score, y_z_score, mode="same")

    # fit the std of a gaussian to the correlation function
    def loss(p):
        return p[0] * norm.pdf(x, 0, p[1]) - y_corr

    p_init = np.array((np.max(y_corr), 1.0))
    p_hat = optimize.leastsq(loss, p_init)[0]

    # return the standard deviation
    return p_hat[1]


def _broadcast_per_neuron(x, n_neurons, name):
    """
    x can be scalar or (n_neurons,) array-like. Returns (n_neurons,) np.ndarray.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return np.full((n_neurons,), float(x), dtype=float)
    if x.ndim == 1 and x.shape[0] == n_neurons:
        return x
    raise ValueError(
        f"{name} must be a scalar or shape (n_neurons,), got shape {x.shape}"
    )


def tmac_ac_fixed_timescales(
    red_np,
    green_np,
    tau_a,
    tau_m,
    optimizer="BFGS",
    verbose=False,
    truncate_freq=True,
):
    """
    Same as tmac_ac, except tau_a and tau_m are FIXED (user-provided)
    and we only optimize:
      log(variance_r_noise), log(variance_g_noise), log(variance_a), log(variance_m)

    Parameters
    ----------
    tau_a, tau_m : float or array (n_neurons,)
        Temporal length scales (same units as your time index: typically "samples").
        If you have tau in seconds, convert beforehand: tau_samples = tau_seconds * sample_rate.
    """
    device = "cpu"
    dtype = torch.float64

    red_np = pp.check_input_format(red_np)
    green_np = pp.check_input_format(green_np)

    if np.any(~np.isfinite(red_np)) or np.any(~np.isfinite(green_np)):
        raise Exception("Input data cannot have any nan or inf")
    if red_np.shape != green_np.shape:
        raise Exception("red and green matrices must be the same shape")

    T, N = red_np.shape

    # Broadcast taus per neuron
    tau_a = _broadcast_per_neuron(tau_a, N, "tau_a")
    tau_m = _broadcast_per_neuron(tau_m, N, "tau_m")
    if np.any(tau_a <= 0) or np.any(tau_m <= 0):
        raise ValueError("tau_a and tau_m must be > 0")

    # Preprocess to fold-change from mean (same as original)
    mean_red = np.mean(red_np, axis=0)
    mean_green = np.mean(green_np, axis=0)
    red_np = red_np / mean_red - 1
    green_np = green_np / mean_green - 1

    red = torch.tensor(red_np, device=device, dtype=dtype)
    green = torch.tensor(green_np, device=device, dtype=dtype)

    red_fft = tfo.real_fft(red)
    green_fft = tfo.real_fft(green)

    # Initial variances (same style as original)
    variance_r_noise_init = np.var(red_np, axis=0)
    variance_g_noise_init = np.var(green_np, axis=0)
    variance_a_init = np.var(green_np, axis=0)
    variance_m_init = np.var(red_np, axis=0)

    a_trained = np.zeros((T, N), dtype=float)
    m_trained = np.zeros((T, N), dtype=float)

    variance_r_noise_trained = np.zeros((N,), dtype=float)
    variance_g_noise_trained = np.zeros((N,), dtype=float)
    variance_a_trained = np.zeros((N,), dtype=float)
    variance_m_trained = np.zeros((N,), dtype=float)

    # Store the fixed taus too (returned for bookkeeping)
    tau_a_trained = tau_a.copy()
    tau_m_trained = tau_m.copy()

    start = time.time()

    for n in range(N):
        log_tau_a_fixed = torch.tensor(np.log(tau_a[n]), device=device, dtype=dtype)
        log_tau_m_fixed = torch.tensor(np.log(tau_m[n]), device=device, dtype=dtype)

        # Train only 4 params now
        evidence_training_variables = np.log(
            [
                variance_r_noise_init[n],
                variance_g_noise_init[n],
                variance_a_init[n],
                variance_m_init[n],
            ]
        )

        def evidence_loss_fn(training_variables_4):
            # training_variables_4 = [log_var_r_noise, log_var_g_noise, log_var_a, log_var_m]
            return -tpd.tmac_evidence_and_posterior_fixed_taus(
                r=red[:, n],
                fourier_r=red_fft[:, n],
                log_variance_r_noise=training_variables_4[0],
                g=green[:, n],
                fourier_g=green_fft[:, n],
                log_variance_g_noise=training_variables_4[1],
                log_variance_a=training_variables_4[2],
                log_tau_a_fixed=log_tau_a_fixed,
                log_variance_m=training_variables_4[3],
                log_tau_m_fixed=log_tau_m_fixed,
                calculate_posterior=False,
                truncate_freq=truncate_freq,
            )

        trained = opt.scipy_minimize_with_grad(
            evidence_loss_fn,
            evidence_training_variables,
            optimizer=optimizer,
            device=device,
            dtype=dtype,
        )

        trained_torch = torch.tensor(trained.x, dtype=dtype, device=device)

        a, m = tpd.tmac_evidence_and_posterior_fixed_taus(
            r=red[:, n],
            fourier_r=red_fft[:, n],
            log_variance_r_noise=trained_torch[0],
            g=green[:, n],
            fourier_g=green_fft[:, n],
            log_variance_g_noise=trained_torch[1],
            log_variance_a=trained_torch[2],
            log_tau_a_fixed=log_tau_a_fixed,
            log_variance_m=trained_torch[3],
            log_tau_m_fixed=log_tau_m_fixed,
            calculate_posterior=True,
            truncate_freq=truncate_freq,
        )

        a_trained[:, n] = a.detach().cpu().numpy()
        m_trained[:, n] = m.detach().cpu().numpy()

        variance_r_noise_trained[n] = float(torch.exp(trained_torch[0]).cpu().numpy())
        variance_g_noise_trained[n] = float(torch.exp(trained_torch[1]).cpu().numpy())
        variance_a_trained[n] = float(torch.exp(trained_torch[2]).cpu().numpy())
        variance_m_trained[n] = float(torch.exp(trained_torch[3]).cpu().numpy())

        if verbose:
            elapsed = time.time() - start
            remaining = elapsed / (n + 1) * (N - (n + 1))
            print(f"{n + 1}/{N} neurons complete")
            print(f"{elapsed:.3f}s elapsed, estimated {remaining:.3f}s remaining")

    return {
        "a": a_trained,
        "m": m_trained,
        "variance_r_noise": variance_r_noise_trained,
        "variance_g_noise": variance_g_noise_trained,
        "variance_a": variance_a_trained,
        "variance_m": variance_m_trained,
        "length_scale_a": tau_a_trained,  # fixed
        "length_scale_m": tau_m_trained,  # fixed
    }
