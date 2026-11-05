"""Build the student-facing dataset from the raw sensitivity-analysis archive.

This script is for the supervisor only.  It reads the raw CFD sensitivity-analysis
output (read-only) and writes the tidy, documented CSV files under ``data/`` that
the students actually work with.  Students do not need to run it; it is included
so that the published dataset is fully reproducible.

Usage
-----
    python tools/build_dataset.py --raw-channel <dir> --raw-annular-params <dir> \
                                  --raw-annular-cases <dir> --out data

Everything the script reads is treated as read-only.  Nothing is ever written
outside ``--out``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# --------------------------------------------------------------------------
# Problem definition: the nine independent SST k-omega parameters.
# Bounds are the admissible sampling intervals of Table 1 in Eidi & Dwight (2026).
# --------------------------------------------------------------------------
PARAMETERS = [
    # name,          symbol,              description,                                 nominal, lo,     hi
    ("alphaOmega1", r"$\alpha_{\omega1}$", "omega diffusion coefficient (near-wall)",   0.50,   0.30,   0.70),
    ("alphaOmega2", r"$\alpha_{\omega2}$", "omega diffusion coefficient (outer layer)", 0.856,  0.70,   1.00),
    ("alphaK1",     r"$\alpha_{k1}$",      "k diffusion coefficient (near-wall)",       0.85,   0.70,   1.00),
    ("alphaK2",     r"$\alpha_{k2}$",      "k diffusion coefficient (outer layer)",     1.00,   0.80,   1.20),
    ("beta1",       r"$\beta_1$",          "omega destruction coefficient (near-wall)", 0.075,  0.06,   0.09),
    ("beta2",       r"$\beta_2$",          "omega destruction coefficient (outer)",     0.0828, 0.07,   0.10),
    ("betaStar",    r"$\beta^*$",          "k dissipation coefficient",                 0.09,   0.0784, 0.1024),
    ("a1",          r"$a_1$",              "eddy-viscosity limiter coefficient",        0.31,   0.31,   0.40),
    ("kappa",       r"$\kappa$",           "von Karman constant",                       0.41,   0.38,   0.42),
]
PARAM_NAMES = [p[0] for p in PARAMETERS]

# Saltelli design: 1408 = 128 * (9 + 2).  Rows are emitted in blocks of
# (d + 2) = 11, all derived from one base sample, so rows within a block differ
# in at most one parameter.  Splits must keep a block together.
BLOCK_SIZE = len(PARAM_NAMES) + 2
N_BASE = 128
N_CASES = N_BASE * BLOCK_SIZE
N_POINTS = 40

# Reference friction velocity and viscosity used in the raw extraction, for all
# four channel Reynolds numbers (see DATA_DICTIONARY.md).
U_TAU_REF = 0.0128
NU = 2.0e-5

CHANNEL_RE = [180, 395, 640, 1020]

# Raw channel file -> student-facing profile name
CHANNEL_VARIABLES = {
    "p_rms": "p_rms_plus_data.txt",
    "k": "kMean_data.txt",
}

# Raw annular column -> student-facing profile name
# The published analysis (SA_FTaC.py) runs the Sobol estimator on the raw
# pfluctSqrMean column, so we ship exactly that quantity and compute the
# benchmark on it.  See DATA_DICTIONARY.md for what the column means.
ANNULAR_VARIABLES = {
    "p_rms": "pfluctSqrMean",
    "k": "kMean",
}

POINT_COLUMNS = [f"point_{i:02d}" for i in range(N_POINTS)]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def check_saltelli_design(X: np.ndarray, label: str) -> None:
    """Verify the row ordering really is a Saltelli design in blocks of 11."""
    if X.shape != (N_CASES, len(PARAM_NAMES)):
        raise ValueError(f"{label}: expected {(N_CASES, len(PARAM_NAMES))}, got {X.shape}")
    violations = 0
    for b in range(N_BASE):
        base = X[b * BLOCK_SIZE]
        for k in range(len(PARAM_NAMES)):
            changed = np.flatnonzero(X[b * BLOCK_SIZE + 1 + k] != base)
            if changed.tolist() != [k]:
                violations += 1
    if violations:
        raise ValueError(f"{label}: {violations} cross-rows break the Saltelli block structure")
    print(f"  [ok] {label}: Saltelli design verified "
          f"({N_BASE} blocks x {BLOCK_SIZE} rows)")


def write_parameters(raw_csv: Path, out_dir: Path) -> pd.DataFrame:
    """Copy the parameter sample, adding the design_group column."""
    df = pd.read_csv(raw_csv)
    if list(df.columns) != ["case_id"] + PARAM_NAMES:
        raise ValueError(f"{raw_csv}: unexpected columns {list(df.columns)}")
    check_saltelli_design(df[PARAM_NAMES].to_numpy(), raw_csv.parent.name)
    out = df.copy()
    out.insert(1, "design_group", np.arange(len(out)) // BLOCK_SIZE)
    out.to_csv(out_dir / "parameters.csv", index=False)
    return out


def write_profile(values: np.ndarray, case_ids, out_dir: Path, name: str) -> None:
    if values.shape != (N_CASES, N_POINTS):
        raise ValueError(f"{name}: expected {(N_CASES, N_POINTS)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains non-finite values")
    df = pd.DataFrame(values, columns=POINT_COLUMNS)
    df.insert(0, "case_id", case_ids)
    df.to_csv(out_dir / f"profile_{name}.csv", index=False)


def scalar_outputs(profiles: dict[str, np.ndarray], coord: np.ndarray) -> pd.DataFrame:
    """Derive the six scalar quantities of interest from the 40-point profiles."""
    p, k = profiles["p_rms"], profiles["k"]
    return pd.DataFrame({
        "p_rms_peak": p.max(axis=1),
        "p_rms_wall": p[:, 0],
        "p_rms_integral": trapezoid(p, coord, axis=1),
        "k_peak": k.max(axis=1),
        "k_centre": k[:, -1],
        "k_integral": trapezoid(k, coord, axis=1),
    })


SCALAR_NAMES = ["p_rms_peak", "p_rms_wall", "p_rms_integral",
                "k_peak", "k_centre", "k_integral"]


def sobol_benchmark(X_problem: dict, Y: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Compute first- and total-order Sobol indices for a set of outputs.

    Y has one column per output; ``labels`` names them.  Returned in long
    (tidy) format with one row per output-parameter pair.
    """
    from SALib.analyze import sobol

    rows = []
    for j, label in enumerate(labels):
        res = sobol.analyze(X_problem, Y[:, j], calc_second_order=False,
                            print_to_console=False, seed=0)
        for i, pname in enumerate(PARAM_NAMES):
            rows.append({
                "output": label,
                "parameter": pname,
                "S1": res["S1"][i],
                "S1_conf": res["S1_conf"][i],
                "ST": res["ST"][i],
                "ST_conf": res["ST_conf"][i],
            })
    return pd.DataFrame(rows)


def make_problem() -> dict:
    return {
        "num_vars": len(PARAM_NAMES),
        "names": list(PARAM_NAMES),
        "bounds": [[lo, hi] for _, _, _, _, lo, hi in PARAMETERS],
    }


# --------------------------------------------------------------------------
# Channel flow
# --------------------------------------------------------------------------
def build_channel(raw_root: Path, out_root: Path, problem: dict) -> None:
    print("\n=== channel flow ===")
    # The y+ grid is shared by all four Reynolds numbers: the mesh is identical
    # and the raw extraction used one reference u_tau for every case.
    dns = np.loadtxt(raw_root / "SA_results_for_Re_640" / "dns_pRMS_interpolated.txt",
                     skiprows=1)
    y_plus, dns_p_rms = dns[:, 0], dns[:, 1]
    y_metres = y_plus * NU / U_TAU_REF

    channel_dir = out_root / "channel_flow"
    channel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_plus": y_plus, "p_rms_dns": dns_p_rms}).to_csv(
        channel_dir / "dns_reference_p_rms.csv", index=False)

    for re_tau in CHANNEL_RE:
        raw = raw_root / f"SA_results_for_Re_{re_tau}"
        out = channel_dir / f"Re_tau_{re_tau}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"  Re_tau = {re_tau}")

        params = write_parameters(raw / "SA_parameters.csv", out)
        case_ids = params["case_id"].to_numpy()

        profiles = {}
        for name, fname in CHANNEL_VARIABLES.items():
            values = np.loadtxt(raw / fname)
            profiles[name] = values
            write_profile(values, case_ids, out, name)

        pd.DataFrame({"point_index": np.arange(N_POINTS),
                      "y_plus": y_plus,
                      "y_metres": y_metres}).to_csv(
            out / "profile_coordinates.csv", index=False)

        scal = scalar_outputs(profiles, y_plus)
        table = pd.concat([params, scal], axis=1)
        table.to_csv(out / "scalar_outputs.csv", index=False)

        # --- Sobol benchmark -------------------------------------------------
        bench_dir = out_root / "sobol_benchmark"
        bench_dir.mkdir(parents=True, exist_ok=True)

        sobol_benchmark(problem, scal[SCALAR_NAMES].to_numpy(), SCALAR_NAMES).to_csv(
            bench_dir / f"channel_Re_tau_{re_tau}_scalar.csv", index=False)

        prof_frames = []
        for name, values in profiles.items():
            labels = [f"{name}:point_{i:02d}" for i in range(N_POINTS)]
            df = sobol_benchmark(problem, values, labels)
            df["variable"] = name
            df["point_index"] = df["output"].str.split("point_").str[1].astype(int)
            prof_frames.append(df)
        prof = pd.concat(prof_frames, ignore_index=True)
        prof = prof[["variable", "point_index", "parameter", "S1", "S1_conf", "ST", "ST_conf"]]
        prof.to_csv(bench_dir / f"channel_Re_tau_{re_tau}_profiles.csv", index=False)
        print(f"    wrote profiles for {len(CHANNEL_VARIABLES)} variables "
              f"+ {len(SCALAR_NAMES)} scalars + Sobol benchmark")


# --------------------------------------------------------------------------
# Annular flow
# --------------------------------------------------------------------------
def build_annular(raw_params_dir: Path, raw_cases_dir: Path,
                  out_root: Path, problem: dict) -> None:
    print("\n=== annular flow ===")
    out = out_root / "annular_flow"
    out.mkdir(parents=True, exist_ok=True)

    params = write_parameters(raw_params_dir / "SA_parameters.csv", out)
    case_ids = params["case_id"].tolist()

    raw_cols = list(ANNULAR_VARIABLES.values())
    stacked = {c: np.full((N_CASES, N_POINTS), np.nan) for c in raw_cols}
    coord = None
    missing = []
    for i, case_id in enumerate(case_ids):
        f = raw_cases_dir / case_id / "extracted_data.csv"
        if not f.exists():
            missing.append(case_id)
            continue
        df = pd.read_csv(f)
        for c in raw_cols:
            stacked[c][i, :] = df[c].to_numpy()[:N_POINTS]
        if coord is None:
            coord = df["wall_distance"].to_numpy()[:N_POINTS]
        if (i + 1) % 400 == 0:
            print(f"    read {i + 1}/{N_CASES} cases")
    if missing:
        raise RuntimeError(f"missing extracted_data.csv for {len(missing)} cases")

    profiles = {}
    for name, raw_col in ANNULAR_VARIABLES.items():
        values = stacked[raw_col]
        profiles[name] = values
        write_profile(values, case_ids, out, name)

    pd.DataFrame({"point_index": np.arange(N_POINTS),
                  "wall_distance_metres": coord}).to_csv(
        out / "profile_coordinates.csv", index=False)

    scal = scalar_outputs(profiles, coord)
    pd.concat([params, scal], axis=1).to_csv(out / "scalar_outputs.csv", index=False)

    bench_dir = out_root / "sobol_benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    sobol_benchmark(problem, scal[SCALAR_NAMES].to_numpy(), SCALAR_NAMES).to_csv(
        bench_dir / "annular_scalar.csv", index=False)

    prof_frames = []
    for name, values in profiles.items():
        labels = [f"{name}:point_{i:02d}" for i in range(N_POINTS)]
        df = sobol_benchmark(problem, values, labels)
        df["variable"] = name
        df["point_index"] = df["output"].str.split("point_").str[1].astype(int)
        prof_frames.append(df)
    prof = pd.concat(prof_frames, ignore_index=True)
    prof[["variable", "point_index", "parameter", "S1", "S1_conf", "ST", "ST_conf"]].to_csv(
        bench_dir / "annular_profiles.csv", index=False)
    print("    wrote annular profiles + scalars + Sobol benchmark")


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-channel", type=Path, required=True)
    ap.add_argument("--raw-annular-params", type=Path, required=True)
    ap.add_argument("--raw-annular-cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    problem = make_problem()
    args.out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"parameter": n, "symbol": s, "description": d,
          "nominal": nom, "lower_bound": lo, "upper_bound": hi}
         for n, s, d, nom, lo, hi in PARAMETERS]
    ).to_csv(args.out / "turbulence_parameters.csv", index=False)
    print(f"wrote {args.out / 'turbulence_parameters.csv'}")

    build_channel(args.raw_channel, args.out, problem)
    build_annular(args.raw_annular_params, args.raw_annular_cases, args.out, problem)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
