# Finding Important Turbulence-Model Parameters with Explainable AI

**AI Capstone project — TU Delft**
Supervisor: Ali Eidi, Faculty of Aerospace Engineering (`a.eidi@tudelft.nl`)

---

## The problem in one paragraph

Computational Fluid Dynamics predicts turbulent flows using a *turbulence model*.
The model used here, SST k–omega, contains nine empirical constants whose exact
values nobody knows — each is only known to lie inside an interval. Different
choices inside those intervals give different predictions, so an engineer needs
to know **which of the nine actually matter**. The standard way to find out is a
Sobol sensitivity analysis, which here cost **1408 CFD simulations per flow
case**. This project asks whether a machine-learning surrogate plus an
explainability method can reach the same conclusion from far fewer simulations.

The Sobol indices from the full study are provided as a **benchmark**. You never
compute them from CFD — that part is already done. Your job is to see how close
you can get without them.

---

## Getting started

```bash
git clone https://github.com/alieidi1st/tudelft-capstone-ai-cfd-2026.git
cd tudelft-capstone-ai-cfd-2026

# option A - conda (recommended)
conda env create -f environment.yml
conda activate capstone

# option B - pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter lab notebooks/
```

You can clone it anywhere. The notebooks locate the project root themselves, so
nothing depends on where you put the folder, on your username, or on your
operating system.

Then open the notebook for your group. It is a **skeleton, not a worked
example**: the explanatory text, the data loading and the figures that show you
the provided data are given, and the analysis is yours to write. Look for
`# TODO` blocks — each one states the task, gives hints, and names the variables
that later cells expect.

---

## Which notebook is mine?

Both groups share the same dataset, the same benchmark and the same MUST
requirements. They then go in two different and independent directions.

### `notebooks/group_A_data_efficiency.ipynb`
**Can a cheap surrogate replace an expensive sensitivity study?**

Treats the surrogate model as the object of study. Trains several kinds of
surrogate, measures how accuracy and the recovered ranking degrade as you remove
training data, and computes Sobol indices *from the surrogate* to hold against
the indices from all 1408 simulations.

*Closing result:* which surrogate to use, how many CFD simulations it needs, and
whether the expensive study can be replaced by a cheap one.

### `notebooks/group_B_explainability.ipynb`
**Which explanation do we trust, and where in the flow?**

Treats the explanation as the object of study. Fits a surrogate, then applies and
compares several explainability methods, and examines how the ranking depends on
*which* of the 80 CFD outputs you ask about — a single scalar, points along a
profile, or the dominant modes of the whole output field.

*Closing result:* a comparison of explainability methods against the benchmark,
and a map of which parameters matter where.

Neither group needs a result produced by the other.

---

## What is in this repository

```
notebooks/                YOUR WORK GOES HERE - one notebook per group
RESOURCES.md              where to find every method you need
data/                     the dataset (generated; see DATA_DICTIONARY.md)
  turbulence_parameters.csv    the nine parameters, with their sampling ranges
  channel_flow/Re_tau_*/       four Reynolds numbers, 1408 simulations each
  annular_flow/                a second geometry, 1408 simulations
  sobol_benchmark/             the reference answer from the full CFD study
src/capstone_tools.py     loading helpers, parameter metadata, plot style
LICENSE                   MIT for the code and data; the paper stays CC BY 4.0
figures/                  a couple of previews of the provided data
reference/                the paper the data comes from
tools/                    dataset build script (you do not need to run it)
```

**[`RESOURCES.md`](RESOURCES.md)** lists every library function this project
needs, with a documentation link and a worked example for each. You are not
expected to implement any method yourself.

**Read [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) before you start.** It
explains every file, and it contains one warning you must not skip, summarised
next.

---

## The one thing that will catch you out

The 1408 simulations are **not** 1408 independent random samples. They form a
*Saltelli design*: 128 base samples, each expanded into a block of 11 rows, where
9 of those rows differ from the base in exactly one parameter.

So if you call `train_test_split` on the rows, nearly every test simulation has a
near-twin in the training set and your test score will flatter you. Every
`parameters.csv` carries a `design_group` column; keep a whole group on one side
of the split:

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(splitter.split(X, y, groups=data["design_group"]))
```

Both notebooks show the size of the difference in their section 4a. It is a
result worth reporting, not a technicality to hide.

---

## What the full CFD study found

So you know what you are aiming at. Of the nine parameters, only four have any
measurable influence:

| parameter | role |
|---|---|
| **beta\*** | k-dissipation coefficient — dominant in the channel |
| **alpha_omega1** | near-wall omega diffusion — strong second, and dominant in the annular case |
| **kappa** | von Karman constant — small but non-zero |
| **beta1** | near-wall omega destruction — small, mostly near the wall |

The other five are indistinguishable from zero. With increasing Reynolds number
alpha_omega1 grows and beta\* shrinks, until at Re_tau = 1020 they have swapped
places.

Recovering *that* from a few hundred simulations is the project.

---

## Honest expectations

A few things that are true of this dataset and worth knowing early, so you
interpret your own results correctly rather than being surprised by them:

* **Finding the top two parameters is easy.** They carry about 95% of the
  variance. Almost any model with almost any amount of data will find them.
  Getting positions 3–5 right is the hard part.
* **Spearman correlation over all nine parameters is a poor headline measure.**
  Seven of the nine have a benchmark index near zero, so their relative order is
  noise, and that noise does not go away with more data. Top-k overlap is the
  measure that matches the engineering question.
* **A high R² does not guarantee a correct explanation**, and a mediocre R² does
  not guarantee a wrong one. Showing this clearly is a genuine contribution.

---

## Requirements

Python 3.10+, with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy` and
`SALib`. Group B also uses `shap`; Group A optionally uses `xgboost`. Both are
listed in `environment.yml` and `requirements.txt`, and the notebooks degrade
gracefully if they are missing.

## Licence

The code, the notebooks and the derived dataset are released under the **MIT
licence** — see [`LICENSE`](LICENSE). You may use, modify and share them
freely, including for your own coursework.

The article bundled in [`reference/`](reference/) is Open Access under
**CC BY 4.0** and keeps that licence. If you use the dataset in work of your
own, please cite it.

## Source of the data

Eidi, A. & Dwight, R. P. (2026). *A surrogate-based Bayesian framework for
parametric calibration of URANS turbulence models informed by anisotropic
pressure fluctuations.* **Flow, Turbulence and Combustion** 117, 1.
<https://doi.org/10.1007/s10494-026-00763-x>

The paper is in [`reference/`](reference/). You are not reproducing its Bayesian
calibration — you are only using its sensitivity-analysis dataset.
