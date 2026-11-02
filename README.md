# Business Analytics Portfolio

**Ahadi Nur**  
Business Analytics Intern Portfolio | Codveda Technologies

This repository presents a one-month business analytics internship portfolio. It follows the path from reliable data preparation to decision-ready reporting, predictive models, risk prioritization, and strategy recommendations.

## Portfolio at a Glance

| Area | What it demonstrates | Key tools |
| --- | --- | --- |
| [Data Preparation & Exploratory Analysis](data_preparation_and_exploration/README.md) | Data quality, cleaning, feature engineering, and exploratory analysis | Python, pandas, seaborn, matplotlib |
| [Business Intelligence & Decision Support](business_intelligence_and_decision_support/README.md) | Power BI reporting, SQL analysis, statistical inference | Power BI, SQLite, Python, SciPy |
| [Predictive, Risk & Strategy Analytics](predictive_risk_and_strategy_analytics/README.md) | Classification, regression, forecasting, segmentation, risk bands, and scenario planning | scikit-learn, statsmodels, PuLP, Streamlit |

## Selected Business Outcomes

- Built a social-media sentiment reporting asset covering **710 posts**, with **58.9% positive sentiment** and average engagement of **64.5 per post**.
- Analyzed **3,333 customer records** to identify churn concentration among customers with four or more service calls and customers on international plans.
- Evaluated churn relationships with statistical tests and quantified uncertainty using confidence intervals and effect sizes.
- Developed predictive assets for churn, house-price estimation, forecasting, and clustering, then translated churn probabilities into actionable risk bands and allocation scenarios.

## Repository Structure

```text
business-analytics-portfolio/
├── data_preparation_and_exploration/
├── business_intelligence_and_decision_support/
└── predictive_risk_and_strategy_analytics/
```

Each folder contains a focused README, the reproducible code where applicable, and the output artifacts used to support the analysis.

## Notes for Reviewers

- Some source datasets were provided in the internship environment and are not redistributed here. Cleaned datasets and analysis outputs are included where available.
- Python scripts retain environment-specific source paths; update those paths before rerunning.
- Saved model binaries (`.joblib`) and the local SQLite database (`.db`) are intentionally excluded from Git through `.gitignore` to keep the public repository lightweight. The Power BI workbook is included as a portfolio deliverable.

## About

Created by **Ahadi Nur** as part of a Business Analyst internship portfolio at Codveda Technologies.
