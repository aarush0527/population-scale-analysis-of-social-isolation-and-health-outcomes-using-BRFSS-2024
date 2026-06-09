# Social Isolation and Health Outcomes — Causal Inference Study Using BRFSS 2024

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Data: BRFSS 2024](https://img.shields.io/badge/data-BRFSS%202024-green.svg)](https://www.cdc.gov/brfss/annual_data/annual_2024.html)
[![Methods: AIPW · IPTW · MICE](https://img.shields.io/badge/methods-AIPW%20·%20IPTW%20·%20MICE-orange.svg)](#methods)

A six-stage causal epidemiology pipeline that estimates the effect of social isolation on **depression**, **chronic disease multimorbidity**, and **poor self-rated health** in a nationally representative sample of ~200,000 US adults. The study uses the full CDC BRFSS 2024 survey, propensity score weighting, doubly robust AIPW estimation, multiple imputation, and formal sensitivity analysis — implemented end-to-end in a single reproducible Jupyter notebook.

---

## Table of Contents

- [Research Question](#research-question)
- [Data](#data)
- [Pipeline Overview](#pipeline-overview)
- [Methods](#methods)
- [Key Variables](#key-variables)
- [Outputs](#outputs)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Limitations & Assumptions](#limitations--assumptions)
- [References](#references)

---

## Research Question

> **Does social isolation causally increase the risk of depression, chronic disease multimorbidity, and poor health in U.S. adults, after rigorous adjustment for sociodemographic, behavioral, and healthcare access confounders?**

The primary exposure is a composite **Social Isolation Index (SII)** constructed from three BRFSS items: loneliness frequency, adequacy of emotional support, and life satisfaction. Respondents in the top quartile of the SII are classified as "socially isolated." Three outcomes are examined:

| Outcome | Variable | Measure |
|---|---|---|
| Depression | `depression_bin` | Ever diagnosed with depression (binary) |
| Multimorbidity | `multimorbidity` | ≥ 2 concurrent chronic conditions (binary) |
| Poor/fair health | `poor_health_bin` | Self-rated health = fair or poor (binary) |

---

## Data

**Source:** [CDC Behavioral Risk Factor Surveillance System (BRFSS) 2024](https://www.cdc.gov/brfss/annual_data/annual_2024.html)

The BRFSS is the largest ongoing telephone health survey system in the world, conducted annually by the CDC in collaboration with all U.S. states and territories.

| Property | Value |
|---|---|
| Raw sample size | ~457,670 respondents |
| Analytic sample (post-inclusion) | ~199,740 respondents |
| Variables in raw file | 345 |
| Variables selected for analysis | 60+ |
| Survey design | Complex stratified cluster sample |
| Representativeness | Nationally representative (non-institutionalized U.S. adults) |

The dataset is not included in this repository. It is publicly available from the CDC at no cost:

```bash
# Files downloaded automatically by Stage 1
wget -O brfss_data.zip https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip
wget -O var_layout.html https://www.cdc.gov/brfss/annual_data/2024/llcp_varlayout_24_onecolumn.html
wget -O codebook.zip https://www.cdc.gov/brfss/annual_data/2024/zip/codebook24_llcp-v2-508.zip
```

---

## Pipeline Overview

The notebook is organized into six sequential stages, each saving intermediate outputs to disk.

```
Stage 1 ── Data loading, variable selection, recoding, missingness assessment
    │
Stage 2 ── Multiple imputation (MICE), survey-weighted Table 1, EDA
    │
Stage 3 ── DAG-informed propensity score model, stabilized IPTW, balance diagnostics
    │
Stage 4 ── IPTW outcome regression + AIPW doubly robust estimation (Rubin's rules)
    │
Stage 5 ── E-value sensitivity analysis, Rosenbaum bounds, causal mediation
    │
Stage 6 ── Subgroup/effect modification analysis, final publication-ready figures
```

---

## Methods

### Exposure Construction — Social Isolation Index (SII)

Three BRFSS items are reverse-coded so that higher scores indicate greater isolation/worse social connectedness, z-score standardized, and averaged into a composite SII. Respondents in the top quartile (SII ≥ 75th percentile) are classified as `isolated = 1`.

### Stage 1 — Data Preparation

- Selective import of 60+ variables from the raw XPT file
- BRFSS-specific missing/refusal code handling (`7`, `9`, `77`, `99`, etc.)
- Binary and ordinal recoding of all exposure, outcome, and confounder variables
- Chronic disease multimorbidity index: sum of 10 binary conditions (heart attack, coronary HD, stroke, asthma, skin cancer, other cancer, COPD, kidney disease, arthritis, diabetes)
- ACE score: count of 9 adverse childhood experience indicators
- Analytic sample restriction: non-zero survey weight + non-missing exposure and at least one outcome

### Stage 2 — Multiple Imputation (MICE)

Missing confounder data is handled via **Multiple Imputation by Chained Equations** using `miceforest` (Random Forest backend):
- **m = 5** imputed datasets
- **5 iterations** with convergence tracing
- Variables with > 40% missingness excluded from imputation
- Survey-weighted **Table 1** built with custom weighted proportion and standardized mean difference (SMD) functions
- Unadjusted Love plot for baseline covariate imbalance

### Stage 3 — Propensity Score & IPTW

A **logistic regression propensity score model** is fit on all confounders plus three DAG-motivated interaction terms:

| Interaction | Rationale |
|---|---|
| Age × Marital status | Social isolation pathway differs across life course and partnership status |
| Age × Employment | Labor force exit is a major isolation mechanism in older adults |
| Age × Insurance | Healthcare-mediated pathways differ by age |

**Stabilized IPTW weights** are computed using marginal treatment probabilities as the numerator, trimmed at the 1st/99th percentile, and multiplied by the BRFSS survey weight to maintain national representativeness. Effective sample sizes (ESS) and post-IPTW SMDs are reported.

### Stage 4 — Causal Effect Estimation

Two estimators are applied to all five imputed datasets and pooled using **Rubin's (1987) combining rules**:

**A. IPTW-Weighted Logistic Regression**
Outcome model: `logit(Y) ~ A + L` with combined `IPTW × survey_weight`. Provides doubly-weighted estimates with confounder adjustment both via weighting and regression.

**B. AIPW Doubly Robust Estimator**

The AIPW influence function:

$$\hat{\tau}_{AIPW} = \frac{1}{n} \sum_{i} \left[ \frac{A_i}{\hat{\pi}(L_i)} (Y_i - \hat{\mu}_1(L_i)) - \frac{1-A_i}{1-\hat{\pi}(L_i)} (Y_i - \hat{\mu}_0(L_i)) + \hat{\mu}_1(L_i) - \hat{\mu}_0(L_i) \right]$$

- Outcome model: **Gradient Boosting Classifier** with 5-fold cross-fitting to avoid overfitting bias
- Standard errors from the empirical influence function (sandwich estimator)
- Consistent if *either* the propensity score model or the outcome model is correctly specified

Results are reported as odds ratios (OR), absolute risk differences (ATE in percentage points), 95% confidence intervals, and fraction of missing information (FMI).

### Stage 5 — Sensitivity Analysis

**E-values (VanderWeele & Ding 2017)**
Quantify the minimum strength of association an unmeasured confounder would need with *both* the exposure and outcome to fully explain away the observed effect. Computed on the risk ratio scale via the Zhang & Yu (1998) OR-to-RR conversion.

**Rosenbaum Γ bounds**
Approximate the sensitivity of results to unmeasured binary confounding by computing upper-bound p-values across Γ ∈ [1.0, 3.0] using the normal approximation to the signed rank statistic. The critical Γ is also derived analytically.

**Causal Mediation Analysis**
Decomposes the total effect of social isolation on **multimorbidity** into:
- **Natural Direct Effect (NDE):** effect not passing through depression
- **Natural Indirect Effect (NIE):** effect operating via the depression pathway
- **Proportion Mediated (PM)**

Implemented via Monte Carlo counterfactual integration (200 draws per person) with bootstrap standard errors (B = 500 resamples).

### Stage 6 — Subgroup & Effect Modification Analysis

IPTW-weighted logistic regression is repeated within strata of five pre-specified effect modifiers: sex, age group (18–44 / 45–64 / 65+), race/ethnicity, education, and employment status. Formal interaction tests use likelihood ratio tests (LRT) comparing main-effects vs. interaction models. Subgroup forest plots are produced for all three outcomes.

---

## Key Variables

### Exposure
| Variable | BRFSS Item | Description |
|---|---|---|
| `SDLONELY` | Loneliness | How often feel lonely (1=always → 5=never) |
| `EMTSUPRT` | Emotional support | How often get needed support (1=always → 5=never) |
| `LSATISFY` | Life satisfaction | Overall life satisfaction (1=very satisfied → 4=very dissatisfied) |

### Outcomes
| Variable | BRFSS Item | Description |
|---|---|---|
| `ADDEPEV3` | Depression diagnosis | Ever told by doctor had depression |
| 10 chronic disease items | See codebook | Heart attack, stroke, COPD, diabetes, etc. |
| `GENHLTH` | General health | Self-rated overall health |

### Confounders (selected)
Demographics, socioeconomic status (income, education, employment, marital status, housing), health behaviors (smoking, physical activity, BMI, alcohol), healthcare access (insurance, personal doctor, cost barrier, routine checkup), social determinants of health (food stamps, financial hardship, neighborhood safety), and adverse childhood experiences (ACE score, 9 items).

---

## Outputs

| File | Contents |
|---|---|
| `brfss_analytic_stage1.parquet` | Recoded analytic dataset |
| `brfss_analytic_stage2.parquet` | Post-MICE primary imputed dataset |
| `brfss_analytic_stage3.parquet` | Dataset enriched with propensity scores and IPTW weights |
| `table1_weighted.csv` | Survey-weighted Table 1 by isolation status with SMDs |
| `table2_causal_estimates.csv` | Pooled IPTW and AIPW results (OR, ATE, 95% CI, FMI) |
| `smd_balance.csv` | Pre- and post-IPTW standardized mean differences |
| `ps_model_coefficients.csv` | Propensity score model coefficients and odds ratios |
| `evalues.csv` | E-values for each outcome |
| `rosenbaum_bounds.csv` | Sensitivity p-values across Γ range |
| `mediation_results.csv` | TE, NDE, NIE, PM with bootstrap CIs |
| `subgroup_results.csv` | Subgroup ORs, RDs, and LRT p-values |
| `fig_missingness.png` | Missingness bar chart and per-respondent distribution |
| `fig_mice_convergence.png` | MICE trace plots (mean of imputed values × 5 datasets × 5 iters) |
| `fig_eda.png` | SII distribution, outcome prevalence, correlations |
| `fig_love_plot_unadjusted.png` | Unadjusted SMD Love plot |
| `fig_ps_diagnostics.png` | PS overlap, IPTW distribution, calibration, post-IPTW Love plot |
| `fig_forest_plot.png` | OR and ATE forest plot (crude vs. IPTW vs. AIPW) |
| `fig_sensitivity_mediation.png` | E-value bar chart, Rosenbaum curves, mediation diagram |
| `fig_subgroup_forest.png` | Subgroup forest plots by sex, age, race, education, employment |

---

## Getting Started

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy pyreadstat missingno \
            miceforest tableone scikit-learn statsmodels
```

A machine with at least **16 GB RAM** is recommended. Loading the raw BRFSS XPT file (~1.5 GB) and running MICE + GBM cross-fitting are the most memory-intensive steps. Each stage clears large objects from memory before proceeding.

### Running the notebook

```bash
git clone https://github.com/aarush0527/population-scale-analysis-of-social-isolation-and-health-outcomes-using-BRFSS-2024.git
cd population-scale-analysis-of-social-isolation-and-health-outcomes-using-BRFSS-2024
jupyter notebook Social_Isolation_and_Health_Outcomes_—_Population_Scale_Analysis_Using_BRFSS_2024.ipynb
```

Run stages sequentially. Each stage reads from the previous stage's `.parquet` or `.pkl` output.

| Stage |
|---|
| Stage 1 — Data loading & recoding |
| Stage 2 — MICE imputation |
| Stage 3 — PS estimation |
| Stage 4 — AIPW (GBM, 5-fold × 5 datasets × 3 outcomes) |
| Stage 5 — Mediation bootstrap (B=500) |
| Stage 6 — Subgroup analysis |

---

## Repository Structure

```
.
├── Social_Isolation_and_Health_Outcomes_—_Population_Scale_Analysis_Using_BRFSS_2024.ipynb              # Main notebook (all 6 stages)
├── README.md
│
├── outputs/                       # Generated on first run (not tracked by git)
    ├── *.parquet                  # Intermediate analytic datasets
    ├── table*.csv                 # Result tables
    ├── fig_*.png                  # Figures
    └── *.csv                      # Diagnostic and sensitivity outputs

```

---

## Limitations & Assumptions

- **Cross-sectional design:** The BRFSS is a cross-sectional survey, so exposure and outcome are measured at the same point in time. The causal ordering (isolation → health outcomes) is theoretically motivated and consistent with the broader literature but cannot be verified from this data alone.
- **Self-reported data:** All variables, including chronic disease diagnoses, are based on self-report and subject to recall and social desirability bias.
- **Unmeasured confounding:** Despite adjusting for 25+ confounders, residual confounding from unmeasured variables (e.g., genetic predisposition, early-life social environment beyond ACE items) remains possible. E-values and Rosenbaum Γ bounds are reported to characterize the robustness of findings.
- **Social Isolation Index:** The SII composite is constructed from three available BRFSS items and is a proxy for the broader construct of social isolation. The top-quartile binary cutoff is analytically tractable but somewhat arbitrary; sensitivity analyses using the continuous SII are recommended.
- **MICE assumption:** Multiple imputation assumes missingness is at random (MAR) conditional on observed variables in the imputation model.
- **Survey weights:** IPTW and survey weights are combined multiplicatively. This preserves national representativeness but may amplify weight variability in small subgroups.

---

## References

- VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. *Annals of Internal Medicine.* 2017;167(4):268–274.
- Rubin DB. *Multiple Imputation for Nonresponse in Surveys.* Wiley; 1987.
- Imai K, Keele L, Tingley D. A general approach to causal mediation analysis. *Psychological Methods.* 2010;15(4):309–334.
- Rosenbaum PR. *Observational Studies.* 2nd ed. Springer; 2002.
- Zhang J, Yu KF. What's the relative risk? A method of correcting the odds ratio in cohort studies of common outcomes. *JAMA.* 1998;280(19):1690–1691.
- CDC. Behavioral Risk Factor Surveillance System 2024 Survey Data and Documentation. Atlanta, GA: Centers for Disease Control and Prevention; 2025.
