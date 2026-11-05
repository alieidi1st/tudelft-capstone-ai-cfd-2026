# Where to find the methods you need

Every method this project asks for already exists in a library. You are not
expected to implement any of them. This page points you straight at the
documentation and at a worked example for each one, so you do not have to go
hunting.

Start with the two in **bold** — they are all the MUST requirements need.

---

## 1. The surrogate model

Predicts a CFD output from the nine turbulence-model parameters.

| what | where |
|---|---|
| **Random Forest — API** | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html |
| **Random Forest — user guide** | https://scikit-learn.org/stable/modules/ensemble.html#random-forests |
| Gradient Boosting — API | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html |
| Gradient Boosting — faster variant for larger data | https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html |
| XGBoost — scikit-learn interface | https://xgboost.readthedocs.io/en/stable/python/python_api.html |
| XGBoost — getting started | https://xgboost.readthedocs.io/en/stable/get_started.html |

A working starting point, which you can paste and then tune:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
```

Gradient boosting is a drop-in replacement:

```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=0)
```

## 2. Splitting the data — read this one

This dataset needs a *grouped* split (see `DATA_DICTIONARY.md` section 3). An
ordinary random split will flatter your model.

| what | where |
|---|---|
| **GroupShuffleSplit — API** | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html |
| GroupKFold — API | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html |
| Cross-validation with groups — user guide | https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data |

## 3. Measuring accuracy

| what | where |
|---|---|
| **Regression metrics (RMSE, R²) — user guide** | https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics |
| `mean_squared_error` | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html |
| `r2_score` | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html |

## 4. Explaining the model

| what | where |
|---|---|
| **Permutation importance — API** | https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html |
| **Permutation importance — user guide** | https://scikit-learn.org/stable/modules/permutation_importance.html |
| Worked example, including the training-vs-test trap | https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html |
| SHAP — documentation | https://shap.readthedocs.io/en/latest/ |
| SHAP — `TreeExplainer`, the fast exact method for forests | https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html |
| SHAP — full API index | https://shap.readthedocs.io/en/latest/api.html |
| Partial dependence and ICE — user guide | https://scikit-learn.org/stable/modules/partial_dependence.html |
| `PartialDependenceDisplay` — API | https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html |

A working starting point:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=30,
                                random_state=0, n_jobs=-1, scoring="r2")
result.importances_mean   # one number per parameter
result.importances_std    # its spread over the repeats
```

## 5. Comparing two rankings

| what | where |
|---|---|
| Spearman rank correlation | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html |
| Kendall's tau | https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html |

## 6. Optional extras

| what | where |
|---|---|
| SALib — sensitivity analysis from a surrogate | https://salib.readthedocs.io/en/latest/ |
| SALib — Sobol analysis API | https://salib.readthedocs.io/en/latest/api.html |
| PCA — API | https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html |
| Gaussian process regression — user guide | https://scikit-learn.org/stable/modules/gaussian_process.html |
| Small neural network (`MLPRegressor`) | https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html |

## 7. Background reading, if you want it

| what | where |
|---|---|
| *Interpretable Machine Learning* — free online book | https://christophm.github.io/interpretable-ml-book/ |
| ... its chapter on permutation importance | https://christophm.github.io/interpretable-ml-book/feature-importance.html |
| ... its chapter on SHAP | https://christophm.github.io/interpretable-ml-book/shap.html |

None of this is required reading. The notebooks are written to be
self-contained.
