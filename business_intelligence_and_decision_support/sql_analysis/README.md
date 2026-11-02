# Customer Churn SQL Analysis

**Author:** Ahadi Nur

## Objective

Use SQL to segment customers, identify churn patterns, and demonstrate database-query optimization.

## Dataset

The local SQLite database contains **3,333 customers**, combining non-overlapping training and test records. `Source_File` preserves the original file designation.

## Key Findings

- Overall churn rate: **14.49%** (483 customers).
- Customers with **4+ service calls** have a **51.7%** churn rate.
- International-plan holders churn at **42.4%**, versus **11.5%** for other customers.
- Churned customers average **206.9** total day minutes, compared with **175.2** for non-churned customers.
- California and New Jersey show the highest state-level churn rate among the reviewed states (**26.5%**).

## Included Files

- `churn_queries.sql` — documented queries using aggregations, CTEs, window functions, `CASE`, joins, and index creation.
- `churn.db` — local SQLite database for demonstration; excluded from Git by default.

## Optimization Evidence

An index on `(State, Churn)` changes the query plan for a Texas churn lookup from a table scan to an indexed search.
