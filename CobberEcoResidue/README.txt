# CobberEcoResidue algal bloom teaching datasets

These are invented, curated teaching datasets for the ML for Ecology error-analysis chapter.
Outcome: chlorophyll a concentration in micrograms per liter (ug/L).
Residual convention: residual = Predicted - Actual.
Months are limited to May through September.

Columns:
- Lake
- Month
- LakeSetting
- StormStatus
- BloomSeverity
- Actual
- Predicted

Dataset purposes:
- good_overall_fit.csv: Reasonable predictions; residuals scatter around zero.
- high_scatter.csv: Noisy predictions; large residuals occur in both directions.
- consistent_overprediction.csv: Predictions tend to be too high; positive bias.
- consistent_underprediction.csv: Predictions tend to be too low; negative bias.
- severe_blooms_underpredicted.csv: Mild/moderate blooms fit fairly well, but severe blooms are underpredicted.
- developed_lakes_underpredicted.csv: Developed shoreline lakes are underpredicted more than isolated shoreline lakes.
- august_blooms_underpredicted.csv: August samples are underpredicted compared with other months.
- post_storm_blooms_missed.csv: After-storm samples are underpredicted compared with before-storm samples.
