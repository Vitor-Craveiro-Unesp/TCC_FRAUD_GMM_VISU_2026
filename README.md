# Visualizing Credit Card Fraud Detection using Gaussian Mixture Models (GMM)

An end-to-end statistical learning and data visualization project focused on credit card fraud detection.

This project was developed in the context of an undergraduate degree in Statistics and investigates fraudulent financial transactions through exploratory data analysis, dimensionality reduction, probabilistic clustering and visual analytics.

The study combines Principal Component Analysis (PCA), t-Distributed Stochastic Neighbor Embedding (t-SNE) and Gaussian Mixture Models (GMM) to explore patterns in highly imbalanced financial data and to investigate whether fraudulent transactions occupy specific regions of the feature space.

---

## Project Motivation

Credit card fraud detection is a challenging statistical problem because fraudulent transactions represent only a very small proportion of all financial transactions.

This imbalance makes traditional analysis difficult and creates several challenges, including:

- severe class imbalance;
- rare anomalous observations;
- overlapping transaction profiles;
- complex relationships between variables;
- high dimensionality;
- non-linear structures;
- heterogeneous behavioral patterns.

Because fraudulent transactions may not form clearly separated groups, this project focuses not only on classification, but also on understanding the latent statistical structure of the data.

The main idea is to investigate whether unsupervised and multivariate statistical methods can reveal meaningful structures associated with fraudulent behavior.

---

## Main Objective

The main objective of this project is to investigate the structure of credit card transaction data and evaluate how dimensionality reduction and probabilistic clustering techniques can support the identification and visualization of fraudulent patterns.

---

## Specific Objectives

The project aims to:

- preprocess and organize a large credit card transaction dataset;
- perform exploratory data analysis;
- investigate the distribution and behavior of transaction-related variables;
- identify potential anomalies and unusual transaction profiles;
- perform feature engineering;
- encode categorical variables appropriately;
- prepare the data for multivariate statistical analysis;
- apply Principal Component Analysis;
- apply t-SNE for non-linear visualization;
- estimate Gaussian Mixture Models;
- identify latent transaction groups;
- analyze probabilistic cluster membership;
- investigate the concentration of fraudulent transactions among clusters;
- compare different low-dimensional representations;
- visually explore anomaly distributions;
- evaluate the statistical usefulness and limitations of the selected methods.

---

## Principal Component Analysis — PCA

Principal Component Analysis is a multivariate statistical technique used to reduce dimensionality while preserving as much information as possible from the original dataset.

PCA transforms the original variables into a new set of orthogonal variables known as principal components.

These components are linear combinations of the original variables and are ordered according to the amount of variance they explain.

Within this project, PCA is intended to support:

- dimensionality reduction;
- visualization of high-dimensional data;
- identification of dominant variability patterns;
- reduction of redundancy between correlated variables;
- investigation of transaction structure;
- preparation of lower-dimensional representations for subsequent analyses.

An important aspect of PCA is the analysis of explained variance.

This allows the project to evaluate how much of the original information can be represented using a smaller number of components.

Component loadings may also be examined to understand which original variables contribute most strongly to each principal component.

---

## t-Distributed Stochastic Neighbor Embedding — t-SNE

t-SNE is a non-linear dimensionality reduction method mainly used for data visualization.

While PCA focuses on preserving global linear variance, t-SNE attempts to preserve local neighborhood relationships between observations.

This characteristic makes t-SNE useful for visually exploring complex structures that may not be visible through linear transformations.

Within this project, t-SNE may help visualize:

- local concentrations of transactions;
- possible anomalous regions;
- groups with similar transaction profiles;
- overlap between legitimate and fraudulent observations;
- structures that may not be evident through PCA;
- relationships between GMM clusters and transaction classes.

t-SNE is used primarily as an exploratory visualization technique.

Its output is therefore interpreted cautiously, since distances and cluster shapes in the final representation should not automatically be interpreted as direct statistical evidence.

---

## Gaussian Mixture Models — GMM

Gaussian Mixture Models are probabilistic models used to represent a dataset as a mixture of multiple Gaussian distributions.

Each Gaussian component represents a latent probabilistic group within the data.

Unlike clustering methods based only on deterministic distance rules, GMM estimates the probability that each observation belongs to each cluster.

This probabilistic characteristic is particularly relevant for financial transaction data because transaction groups may overlap.

A transaction may therefore have partial membership probabilities across multiple latent components.

Within this project, GMM is the central modeling technique.

The method is used to investigate:

- latent transaction profiles;
- probabilistic cluster membership;
- overlapping transaction groups;
- covariance structures;
- low-density observations;
- unusual transaction profiles;
- possible concentrations of fraudulent transactions.

The implemented workflow also estimates a historical fraud risk for each Gaussian component using training labels and soft component responsibilities. GMM fitting itself remains unsupervised; labels are used only in the separate risk-estimation stage. This does not constitute an independent final-test result.

---

## Relationship Between PCA, t-SNE and GMM

The three methods play complementary roles in the project.

PCA provides a linear statistical representation of the data. t-SNE provides a non-linear representation that emphasizes local neighborhood relationships. GMM estimates probabilistic mixture components on a selected low-dimensional representation.

Conceptually, the implemented workflow supports the following alternatives:

```text
Original high-dimensional transaction data
                   |
                   v
       Training-only preprocessing
                   |
                   v
         Feature transformation
                   |
           +-------+-------+
           |               |
           v               v
          PCA             t-SNE
           |               |
           +-------+-------+
                   |
                   v
                 GMM
                   |
                   v
       Soft Gaussian responsibilities
                   |
                   v
  Historical risk estimated from training labels
                   |
                   v
      Transaction-level fraud scores
                   |
                   v
     Out-of-fold model evaluation
```

The diagram represents alternative dimensionality-reduction paths, not simultaneous PCA and t-SNE fitting. The GMM does **not** use fraud labels during its own fit. Historical risk is derived separately from labeled training observations.

---

## Dataset and Feature Preparation

The project's YAML configuration references the [Kaggle fraud-detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data), with the locally prepared Parquet file at `data/raw/dataset_raw.parquet`.

- Target: `TARGET_OMEGA` (`1` = fraud; `0` = legitimate transaction).
- Identifier: `NID_ALPHA`.
- Frequency-based encoding: columns ending with `_FEWF`.
- Binary mappings: columns ending with `_BE`.
- One-hot encoding: columns ending with `_OHEWI`, with unknown categories ignored.
- Cyclic features: `_SIN` and `_COS`, already transformed and not standardized according to the YAML.
- Numerical preprocessing: median imputation and standardization, fitted using training data only.

The source dataset, prepared Parquet, and fold assignments are local artifacts; they are not included in the GitHub repository. Data acquisition and any dataset transformation must be documented and verified separately before claiming full reproduction on another machine.

## Experimental Design and Leakage Prevention

The saved data split uses **10 stratified, shuffled folds** and random seed **42**:

| Partition | Role |
| --- | --- |
| F1–F9 | Development and out-of-fold (OOF) validation |
| F10 | Reserved independent final test; excluded from the documented OOF runs |

For each OOF validation fold among F1–F9, the remaining eight development folds provide the training data. Encoding, imputation, standardization, dimensionality reduction, GMM fitting, and historical-risk estimation are fitted without using the held-out validation fold. The GMM is fitted on a training-only random-under-sampling (RUS) subset, whereas Gaussian historical risks are estimated using the **original training set** and its labels.

The current PCA test configuration uses:

| Parameter | Value for the recent PCA sample runs |
| --- | --- |
| PCA dimensions | 1D, 2D, and 3D, executed separately |
| GMM components | `K = 4` |
| GMM covariance | `full` |
| GMM initializations | `n_init = 5` |
| RUS ratio | 16 legitimate observations per fraud observation |
| Random seed | `42` |
| OOF validation | F1–F9 |
| Quick-run sample | 1,000 observations per development fold (9,000 total) |

These settings describe the **recent exploratory runs**, not an assertion that all past experiments used identical hyperparameters. The main YAML currently holds `PCA` with two components; the three-dimension execution script temporarily applies the 1D, 2D, and 3D configurations and restores the original YAML bytes afterward.

**Important:** a directory or visualization path named `teste_final` does not, by its name alone, establish that an F10 evaluation took place. The experimental manifest and its data-origin evidence must be checked before reporting final-test results.

## Probability Estimation and Evaluation

After fitting the unlabeled Gaussian mixture, the pipeline computes soft component responsibilities. Historical fraud risks are estimated from those responsibilities and the original labels of the corresponding training portion. A transaction-level fraud score is then calculated as a weighted combination of component risks.

The implemented OOF evaluation uses:

- **Average Precision (AP):** precision–recall ranking performance on an imbalanced outcome.
- **Matthews Correlation Coefficient (MCC):** threshold-dependent classification performance.
- **LogLoss:** evaluation of predicted fraud probabilities.
- **Composite score:** the implemented model-selection criterion.

The YAML defines the composite score as:

```text
Composite = [AP + (MCC + 1)/2 + exp(-LogLoss)] / 3
```

The operating threshold `tau*` is selected from OOF predictions to optimize MCC; visual comparisons may also show the reference threshold `tau = 0.50`. The final independent test fold must not participate in candidate selection or threshold tuning.

Sample runs are explicitly marked as exploratory/OOF sample evaluations. They must not be presented as final independent-test performance.

## Visualizations and PRE01 V2

The project generates static figures and local HTML galleries for model diagnostics, PCA/t-SNE representations, Gaussian geometry, scores, decision thresholds, and performance metrics. Some PCA 3D/GMM visualizations are interactive. The exact gallery contents depend on the experiment configuration and completed generation stages.

**PRE01 V2 — Complete Standardization Diagnostics** is an independent local HTML report integrated into the PCA 1D, 2D, and 3D galleries. It provides:

- Means, variances, and standard deviations of processed variables;
- Standardization parameters and post-transform diagnostics across the development folds;
- Diagnostic PCA component stability across folds;
- Detailed tables, interactive charts, and CSV exports;
- Metadata describing the data and fold provenance.

PRE01 V2 computes additional **diagnostic** preprocessing/PCA information from F1–F9. It does not retrain the GMM or rerun OOF prediction generation during this integration step, and it must not be interpreted as an independent F10 assessment.

## Project Layout

```text
TCC_EXPERIMENTAL_V3/
├── src/                 # Statistical pipeline, OOF orchestration, diagnostics, visualizations
├── tests/               # Unit/integration tests and data-dependent verification scripts
├── scripts/
│   ├── execucao/        # Experimental launch scripts
│   ├── git/             # Checkpoint-related scripts
│   └── historico/       # Historical scripts (not assumed disposable)
├── docs/                # Technical and historical documentation
├── passos/              # Methodological stages and audit records
├── data/                # Local source data and persisted folds; excluded from Git
├── outputs/             # Local experiment artifacts and HTML galleries; excluded from Git
├── backups/             # Local backups; excluded from Git
├── .venv/               # Local Python virtual environment; excluded from Git
├── config_TCC_EXPERIMENTAL_V3_atualizado.yaml
├── requirements.txt
└── README.md
```

The current local layout may contain additional historical or temporary folders. The actual working copy and each experiment's saved manifest take precedence over this schematic.

## Environment Setup (Windows / PowerShell)

The documented local environment was successfully exercised with **Python 3.13.3** and the pinned packages in `requirements.txt` on Windows. These pins provide an environment specification, but they do not package the source data, compiled wheels, operating-system dependencies, or experimental outputs.

From the project root, for a **new** environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

For an **existing** `.venv`, do not recreate it. Activation is optional because all commands can reference its interpreter explicitly:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip check
```

The pipeline also requires the locally prepared Parquet dataset and a matching saved fold assignment under `data/splits/`. The OOF orchestrator checks their hashes against the corresponding fold manifest.

## Running the PCA 1D / 2D / 3D Sample Workflow

The following is the existing Windows script for running **new exploratory training/OOF sample experiments**, not a read-only test and not the final F10 evaluation:

```powershell
# Run from TCC_EXPERIMENTAL_V3/; uses .venv\Scripts\python.exe
& ".\scripts\execucao\RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2.ps1" -PorFold 1000
```

It sequentially creates new PCA 1D, 2D, and 3D sample runs, verifies their OOF outputs, integrates PRE01 V2, and writes a consolidated panel in `outputs/`. The script temporarily rewrites the principal YAML during the three runs and restores its original bytes afterward; **do not launch two instances concurrently**.

If PowerShell blocks execution of this reviewed, local, unsigned script, a one-process invocation is possible without changing the machine-wide execution policy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\execucao\RODAR_NOVO_TESTE_PCA_1D_2D_3D_COM_PRE01_V2.ps1" -PorFold 1000
```

Use that exception only for a script whose content you have reviewed and trust. To suppress automatic browser opening, set `TCC_NAO_ABRIR_NAVEGADOR=1` in the shell environment before launching the script.

The run prints three `GALERIA_1D`, `GALERIA_2D`, `GALERIA_3D` paths and a `PAINEL` path. Open the individual gallery `index.html` files in the browser, for example:

```powershell
Invoke-Item -LiteralPath ".\outputs\<experiment_folder>\vencedor\visualizacao\galeria\index.html"
```

Replace `<experiment_folder>` with the actual folder printed by the completed run. The consolidated panel and PRE01 V2 launcher are also written locally under `outputs/`.

For a **read-only configuration/OOF candidate listing**, without model fitting:

```powershell
.\.venv\Scripts\python.exe -m src.orquestracao_oof --planejar
```

## Tests and Scientific Reproducibility

The `tests/` directory includes unit tests and separate verification scripts that may depend on the real dataset. Review the selected test file before executing it. As an example of a narrow test invocation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_gmm.py
```

The project saves experiment-specific configurations, manifests, model-selection metrics, OOF predictions, galleries, and, when enabled, library versions and input hashes. Use these run-specific artifacts to document **which** data, folds, settings, and Python environment produced any reported result.

The Git repository is **not a complete experimental backup**. By design, `data/`, `outputs/`, `backups/`, and `.venv/` are excluded from normal Git additions. Maintain separate, verified local backups of these assets. Do not delete or replace a saved dataset, fold assignment, OOF prediction, or experimental manifest solely because its filename looks historical.

### Current verification scope

As of **29 October 2026**, the reported Windows `.venv` had all 102 pinned packages installed; `pip check` found no broken installed requirements, the main scientific imports succeeded, and the user completed new PCA 1D/2D/3D sample runs with PRE01 V2 galleries opening successfully. This is evidence of a working **sample development workflow**, **not** proof of identical numeric results across historical environments, full official OOF evaluation, or independent F10 performance.

---

## Repository and Research Notes

This repository presents statistical modeling, diagnostic visualization, and methodological checks developed as part of an undergraduate Statistics research project. All performance statements should identify the evaluation partition, sampling protocol, threshold-selection procedure, and corresponding experiment artifact. Exploratory visualizations and geometry computed from development data must not be presented as independent final-test results.
