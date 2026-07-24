# Paper draft

This directory contains an evidence-constrained first draft reconstructed from the repository in `../oldFiles`. It does not introduce external results or citations.

## Research objective

The project explores whether local weather observations are associated with, or can predict, LoRa RSSI, SNR, throughput, and latency. The repository has no formal protocol or preregistered hypothesis, so the draft characterizes the work as retrospective and exploratory.

## Structure

- `paper.tex`: IEEEtran entry point.
- `sections/`: introduction, related work, dataset, methodology, preprocessing, experiments, results, discussion/limitations/future work, and conclusion.
- `tables/`: target summaries, correlations, and holdout model results.
- `figures/`: representative repository-generated figures with traceable source scripts.
- `references.bib`: deliberately empty placeholder pending a verified literature review.

## Datasets

- `combined_lora_log_final.csv` / `.xlsx`: 1,059 LoRa observations, December 17, 2024--January 2, 2025.
- `combined_wilson_log_final.csv` / `.xlsx`: 14,494 weather observations over the same endpoints; 2,432 exact duplicate rows and 1,358 missing cells.
- `merged_lora_weather_final.csv`: 1,058 matched rows and 28 columns.
- `lora_log_snow.csv` and `WilsonHall_snow.csv`: earlier December 17--18 session files.
- `merged_lora_weather.csv`: empty and unused.
- `top_*.csv`: stored feature-combination correlation outputs.
- `rssi_linear_regression_coefficients.csv`: stored ordinary-linear-model coefficients.

## Analysis pipeline

1. Parse the timestamp embedded in each LoRa decoded payload.
2. Select weather observations within ±5 seconds.
3. Average numeric weather fields and take the mode of nonnumeric fields.
4. Calculate Pearson correlations between weather fields/composite averages and each link metric.
5. Evaluate linear, polynomial, random-forest, Lasso, Ridge, and KNN regressors with fixed random holdouts.

## Reproducing figures and metrics

Create an environment with Python, pandas, NumPy, scikit-learn, and Matplotlib. Run scripts from `../oldFiles` because they use relative paths:

```powershell
cd ..\oldFiles
$env:MPLBACKEND = "Agg"
python h.py
python PolyRegression.py
python randomForrest.py
python SNRregressor.py
python throughputRegressor.py
python LatecncyRegressor.py
```

`merging.py` regenerates the merged file. Run it only if overwriting that artifact is intended. Some scripts print Unicode emoji and can fail in a legacy Windows console; set `$env:PYTHONIOENCODING = "utf-8"` if needed. Base filenames such as `LinearRegression_feature_importance.png` are reused by the SNR script and are not target-qualified.

## Compiling

From this directory:

```powershell
pdflatex paper.tex
pdflatex paper.tex
```

An installed IEEEtran class is required. `references.bib` is intentionally empty and is not invoked until references are verified.

## Remaining TODOs

- Supply author, affiliation, email, and acknowledgments.
- Perform and verify the related-work search; add only checked citations.
- Document radio hardware/settings, antenna geometry, site, weather instrument, units, packet schedule, and collection procedure.
- Explain the 16-day span and identify individual collection sessions.
- Determine whether duplicate weather rows are valid repeated measurements or ingestion artifacts.
- Parse semicolon-delimited particle speed/diameter vectors into justified scalar summaries, or remove them.
- Re-evaluate with temporal/session-blocked validation, confidence intervals, and multiple-testing control.
- Resolve the identical rain-absolute and rain-accumulated fields and avoid unit-mixed feature averages.
