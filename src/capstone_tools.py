"""Shared helpers for the turbulence-parameter explainable-AI capstone project.

This module deliberately contains **only infrastructure**: locating and loading
the data files, metadata about the nine turbulence-model parameters, and a
consistent plotting style.  The machine learning and the explainability analysis
belong in the notebooks, where you can see and change them.

All paths are resolved relative to this file, so the project can be cloned to
any location on any operating system and will work unchanged.

Typical use from a notebook (see the setup cell for a location-independent way
of putting ``src`` on the path)::

    import capstone_tools as ct

    data = ct.load_scalar_outputs("channel", re_tau=640)
    bench = ct.load_sobol_scalar("channel", re_tau=640)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DATA_DIR", "PARAMETER_NAMES", "PARAMETER_INFO", "SCALAR_OUTPUTS",
    "PROFILE_VARIABLES", "VARIABLE_LABELS", "PRETTY", "COLOURS", "BLOCK_SIZE",
    "N_DESIGN_GROUPS", "load_parameter_info", "parameter_bounds",
    "make_salib_problem", "load_scalar_outputs", "load_profiles",
    "load_coordinates", "load_sobol_scalar", "load_sobol_profiles",
    "load_dns_reference", "case_label", "set_plot_style", "parameter_colours",
]

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
DATA_DIR = (Path(__file__).resolve().parent.parent / "data")

#: Rows of the Saltelli design that share a ``design_group`` differ in at most
#: one parameter, so they must never be split across train and test.
BLOCK_SIZE = 11
N_DESIGN_GROUPS = 128

# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
PARAMETER_NAMES = [
    "alphaOmega1", "alphaOmega2", "alphaK1", "alphaK2",
    "beta1", "beta2", "betaStar", "a1", "kappa",
]

#: The six scalar quantities of interest derived from the 40-point profiles.
SCALAR_OUTPUTS = [
    "p_rms_peak", "p_rms_wall", "p_rms_integral",
    "k_peak", "k_centre", "k_integral",
]

#: The two profile variables.  Two variables x 40 points = 80 outputs.
PROFILE_VARIABLES = ["p_rms", "k"]

VARIABLE_LABELS = {
    "p_rms": r"$p'_{\mathrm{rms}}$",
    "k": r"$k$",
}

#: Maths symbols for the nine parameters, for axis labels and legends.
PRETTY = {
    "alphaOmega1": r"$\alpha_{\omega 1}$",
    "alphaOmega2": r"$\alpha_{\omega 2}$",
    "alphaK1": r"$\alpha_{k1}$",
    "alphaK2": r"$\alpha_{k2}$",
    "beta1": r"$\beta_{1}$",
    "beta2": r"$\beta_{2}$",
    "betaStar": r"$\beta^{*}$",
    "a1": r"$a_1$",
    "kappa": r"$\kappa$",
}

#: One stable colour per parameter, matching the published figures.
COLOURS = {
    "alphaOmega1": "#1f77b4",
    "alphaOmega2": "#17becf",
    "alphaK1": "#2ca02c",
    "alphaK2": "#9467bd",
    "beta1": "#7f7f7f",
    "beta2": "#8c564b",
    "betaStar": "#d62728",
    "a1": "#e377c2",
    "kappa": "#bcbd22",
}

PARAMETER_INFO = None  # filled lazily by load_parameter_info()


# --------------------------------------------------------------------------
# Case resolution
# --------------------------------------------------------------------------
def _case_dir(case: str, re_tau: int | None = None) -> Path:
    """Return the directory for a flow case.

    ``case`` is ``"channel"`` (then ``re_tau`` must be one of 180, 395, 640,
    1020) or ``"annular"``.
    """
    case = case.lower()
    if case in ("channel", "channel_flow"):
        if re_tau is None:
            raise ValueError("re_tau is required for the channel-flow case "
                             "(one of 180, 395, 640, 1020)")
        d = DATA_DIR / "channel_flow" / f"Re_tau_{int(re_tau)}"
    elif case in ("annular", "annular_flow"):
        d = DATA_DIR / "annular_flow"
    else:
        raise ValueError(f"unknown case {case!r}; use 'channel' or 'annular'")
    if not d.is_dir():
        raise FileNotFoundError(f"no such case directory: {d}")
    return d


def case_label(case: str, re_tau: int | None = None) -> str:
    """Human-readable name, for figure titles."""
    if case.lower().startswith("channel"):
        return f"channel flow, $Re_\\tau = {int(re_tau)}$"
    return "annular flow"


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_parameter_info() -> pd.DataFrame:
    """The nine parameters with descriptions, nominal values and bounds."""
    global PARAMETER_INFO
    if PARAMETER_INFO is None:
        PARAMETER_INFO = pd.read_csv(DATA_DIR / "turbulence_parameters.csv")
    return PARAMETER_INFO.copy()


def parameter_bounds() -> list[list[float]]:
    """Sampling intervals, in the order of :data:`PARAMETER_NAMES`."""
    info = load_parameter_info().set_index("parameter")
    return [[float(info.loc[n, "lower_bound"]), float(info.loc[n, "upper_bound"])]
            for n in PARAMETER_NAMES]


def make_salib_problem() -> dict:
    """The SALib ``problem`` dictionary for the nine-parameter space."""
    return {"num_vars": len(PARAMETER_NAMES),
            "names": list(PARAMETER_NAMES),
            "bounds": parameter_bounds()}


def load_scalar_outputs(case: str, re_tau: int | None = None) -> pd.DataFrame:
    """Parameters, design group and the six scalar outputs, one row per simulation."""
    return pd.read_csv(_case_dir(case, re_tau) / "scalar_outputs.csv")


def load_profiles(case: str, variable: str,
                  re_tau: int | None = None) -> pd.DataFrame:
    """The 1408 x 40 profile table for one variable (case_id + point_00..point_39)."""
    if variable not in PROFILE_VARIABLES:
        raise ValueError(f"variable must be one of {PROFILE_VARIABLES}")
    return pd.read_csv(_case_dir(case, re_tau) / f"profile_{variable}.csv")


def load_coordinates(case: str, re_tau: int | None = None) -> pd.DataFrame:
    """Spatial coordinate of each of the 40 profile points."""
    return pd.read_csv(_case_dir(case, re_tau) / "profile_coordinates.csv")


def _bench(name: str) -> pd.DataFrame:
    f = DATA_DIR / "sobol_benchmark" / name
    if not f.is_file():
        raise FileNotFoundError(f"no such benchmark file: {f}")
    return pd.read_csv(f)


def load_sobol_scalar(case: str, re_tau: int | None = None) -> pd.DataFrame:
    """Benchmark Sobol indices for the six scalar outputs (long format)."""
    if case.lower().startswith("channel"):
        return _bench(f"channel_Re_tau_{int(re_tau)}_scalar.csv")
    return _bench("annular_scalar.csv")


def load_sobol_profiles(case: str, re_tau: int | None = None) -> pd.DataFrame:
    """Benchmark Sobol indices at all 40 points of all four variables."""
    if case.lower().startswith("channel"):
        return _bench(f"channel_Re_tau_{int(re_tau)}_profiles.csv")
    return _bench("annular_profiles.csv")


def load_dns_reference() -> pd.DataFrame:
    """DNS pressure-fluctuation reference for the channel (validation only)."""
    return pd.read_csv(DATA_DIR / "channel_flow" / "dns_reference_p_rms.csv")


# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
def set_plot_style() -> None:
    """Apply a consistent, readable matplotlib style for the whole project."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.autolayout": False,
    })


def parameter_colours(names=None) -> list[str]:
    """Colours in the given parameter order (defaults to all nine)."""
    return [COLOURS[n] for n in (names if names is not None else PARAMETER_NAMES)]
