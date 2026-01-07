"""High-level entry point for the TMAC package.

This package exposes the core inference routine, supporting utilities, and 
helper modules used for synthetic data creation and preprocessing of two- 
channel imaging data.
"""

from . import fourier  # expose the Fourier helpers so they can be imported directly
from . import models
from . import optimization
from . import preprocessing
from . import probability_distributions
from . import synthetic_data

from .fourier import get_fourier_basis, get_fourier_freq, real_fft, real_ifft
from .models import initialize_length_scale, tmac_ac
from .optimization import scipy_minimize_with_grad
from .preprocessing import check_input_format, interpolate_over_nans, photobleach_correction
from .probability_distributions import tmac_evidence_and_posterior
from .synthetic_data import col_corr, generate_synthetic_data, ratio_model, softplus

__all__ = [
    "fourier",
    "models",
    "optimization",
    "probability_distributions",
    "preprocessing",
    "synthetic_data",
    "get_fourier_basis",
    "get_fourier_freq",
    "real_fft",
    "real_ifft",
    "initialize_length_scale",
    "tmac_ac",
    "scipy_minimize_with_grad",
    "check_input_format",
    "interpolate_over_nans",
    "photobleach_correction",
    "tmac_evidence_and_posterior",
    "generate_synthetic_data",
    "col_corr",
    "ratio_model",
    "softplus",
]

__version__ = "0.1.0"