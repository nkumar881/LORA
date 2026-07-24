# Repository audit

## Scope examined

The audit recursively examined all substantive repository files, excluding only `.git` and IDE metadata:

- Python (12): `CorrelationFinal.py`, `h.py`, `LatecncyRegressor.py`, `LinearRegression.py`, `main.py`, `merging.py`, `PolyRegression.py`, `randomForrest.py`, `RSSICorrelation.py`, `SNRregressor.py`, `throughputRegressor.py`, `wilsonsnow.py`.
- Data (18 CSV, two XLSX): both consolidated logs/workbooks, both snow-session logs, both merged CSVs, the coefficient table, and all 11 `top_*.csv` result tables.
- Other evidence: `backup.txt` and all 20 PNG files listed below.
- No notebooks, Markdown/README documentation, PDFs, LaTeX, reports, presentations, or saved experiment logs existed before this draft.

## Data and processing findings

- LoRa consolidated data: 1,059 rows × 6 columns, no missing cells or exact duplicate rows.
- Weather consolidated data: 14,494 rows × 22 columns, 1,358 missing cells and 2,432 exact duplicate rows.
- Final merged data: 1,058 rows × 28 columns, 92 missing cells confined to non-scalar/categorical fields; all modeled scalar fields and four targets are complete.
- Merge method: decoded-payload timestamp; weather window ±5 seconds; numeric mean; nonnumeric mode; discard unmatched LoRa rows.
- One LoRa row was unmatched.
- `merged_lora_weather.csv` is zero bytes.
- `AverageParticleSpeed` and `VolumeEquivalentDiameter` have zero scalar-numeric values after coercion.

## Analyses discovered

- Individual Pearson correlations for RSSI, SNR, throughput, and latency.
- Unstandardized arithmetic averages of 2–10 weather fields followed by Pearson correlation.
- Ordinary linear, degree-two polynomial, random-forest, Lasso, Ridge, and KNN regression.
- Fixed random holdouts (`random_state=42`), usually 80/20; RSSI linear regression uses 90/10.
- Metrics: Pearson \(r\), squared correlation, holdout \(R^2\), MSE, RMSE, linear/regularized coefficients, and random-forest impurity importances.

## Figures discovered and provenance

| Figure(s) | Generator | Content |
|---|---|---|
| `weather_metrics_plot.png` | `wilsonsnow.py` | Six weather variables over elapsed time for the snow-session file. |
| `knn_rssi_k3.png`, `k5`, `k7`, `k9` | `h.py` | Actual vs. predicted RSSI for four KNN neighborhood sizes. |
| `poly_regression_rssi_plot.png` | `PolyRegression.py` | Actual vs. predicted RSSI for degree-two polynomial regression. |
| `rf_regression_rssi_plot.png` | `randomForrest.py` | Actual vs. predicted RSSI for a 200-tree forest. |
| `rf_feature_importance.png` | `randomForrest.py` | RSSI random-forest feature importances. |
| `LinearRegression_feature_importance.png`, `Lasso_feature_importance.png`, `Ridge_feature_importance.png`, `RandomForest_feature_importance.png` | `SNRregressor.py` | SNR coefficients/importances; filenames omit the target and may be overwritten. |
| `throughput_feature_importance_*.png` (4) | `throughputRegressor.py` | Throughput coefficients/importances. |
| `latency_feature_importance_*.png` (4) | `LatecncyRegressor.py` | Latency coefficients/importances. |

`main.py` creates four LoRa time-series panels interactively but never saves them. The filename it expects (`lora_log_snow(in).csv`) differs from the stored file, so regeneration requires correcting the path. `wilsonsnow.py` has the analogous `(in)` filename mismatch.

## Quantitative findings used in the paper

- Target summaries and all pairwise correlations appear in the paper tables/text.
- Strongest scalar correlations: rain absolute/accumulated with RSSI \(r=0.385690\), SNR \(r=0.450205\), throughput \(r=0.312413\); wind direction with latency \(r=-0.083630\).
- Best tested holdout scores: RSSI polynomial \(R^2=0.190216\); SNR Lasso \(R^2=0.207761\); throughput Lasso \(R^2=0.117741\); latency Lasso \(R^2=-0.0013\).
- Stored best trios: RSSI \(r=0.385660\), SNR \(r=0.450067\), throughput \(r=0.314373\), latency \(r=-0.097258\).

## Assumptions explicitly avoided

- No radio/weather hardware, frequency, radio settings, geometry, units, or sampling schedule was inferred.
- No causal claim was made.
- No citation was invented.
- No p-value, confidence interval, or unrecorded model result was supplied.
- Vector-valued particle fields were not represented as valid scalar predictors.
- Duplicate weather rows were not assumed to be either errors or valid repetitions.

## Remaining TODOs

See `README_PAPER.md`. The highest-priority items are experimental metadata, verified related work, duplicate-row adjudication, vector-field parsing, and session-blocked re-evaluation.
