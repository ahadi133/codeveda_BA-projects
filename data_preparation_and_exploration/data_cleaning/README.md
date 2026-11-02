# Data Cleaning: Sentiment & Iris Datasets

**Author:** Ahadi Nur

## Objective

Create consistent, analysis-ready datasets for later reporting, statistical analysis, and modeling.

## Work Completed

- Removed redundant source index columns and duplicate social-media posts.
- Standardized text fields, platform names, and country labels.
- Parsed timestamps and numeric engagement fields into usable types.
- Filled blank hashtags with `No Hashtag` and missing categorical values with `Unknown`.
- Consolidated detailed emotion labels into Positive, Negative, and Neutral sentiment groups.
- Standardized Iris column names and verified completeness.

## Outputs

- `sentiment_cleaned.csv` — 710 cleaned social-media records.
- `iris_cleaned.csv` — 150 cleaned Iris observations.

## Why It Matters

The cleaned sentiment data becomes the single source for the dashboard and reporting work; reliable type and category definitions prevent misleading downstream metrics.
