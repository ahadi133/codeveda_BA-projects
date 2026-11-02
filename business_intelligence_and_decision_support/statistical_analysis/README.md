# Customer Churn Statistical Analysis

**Author:** Ahadi Nur

## Objective

Quantify the evidence behind churn patterns and separate statistically robust findings from causal claims.

## Findings

- Churned customers make **0.78** more service calls on average than non-churned customers (Welch t-test, p < 0.001; Cohen's d = 0.50).
- International-plan status is strongly associated with churn (chi-square p < 0.001; Cramér's V = 0.26).
- Estimated churn rate: **14.49%**, with a 95% Wilson confidence interval of **13.34%–15.73%**.
- No randomized campaign variable was available, so the analysis does not make causal A/B claims.

## Recommended Actions

1. Prioritize customers with four or more service calls for retention outreach.
2. Investigate price, service quality, and expectation fit for international-plan customers.
3. Validate future retention offers with a randomized, 90-day A/B experiment before scaling.

## Included Artifact

`stats_charts.png` visualizes service-call behavior, churn rates, and a Poisson comparison.
