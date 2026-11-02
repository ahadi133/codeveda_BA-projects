-- Schema: customers(Customer_ID, State, ..., Churn, Source_File) ; state_region(State, Region)

-- q1_count
SELECT COUNT(*) AS total_customers FROM customers;

-- q2_churn_dist
SELECT Churn, COUNT(*) AS customer_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM customers),2) AS pct FROM customers GROUP BY Churn;

-- q3_avg_by_churn
SELECT Churn, COUNT(*) AS n, ROUND(AVG("Total day minutes"),2) AS avg_day_min,
   ROUND(AVG("Customer service calls"),2) AS avg_service_calls,
   ROUND(100.0*SUM("International plan"='Yes')/COUNT(*),1) AS pct_intl_plan,
   ROUND(100.0*SUM("Voice mail plan"='Yes')/COUNT(*),1) AS pct_vmail_plan FROM customers GROUP BY Churn;

-- q4_state_service_calls
SELECT State, COUNT(*) AS customers, ROUND(AVG("Customer service calls"),2) AS avg_service_calls
   FROM customers GROUP BY State HAVING COUNT(*)>=10 ORDER BY avg_service_calls DESC LIMIT 5;

-- q5_cte_state_churn_rank
WITH s AS (SELECT State, COUNT(*) n, SUM(Churn) churned, 100.0*SUM(Churn)/COUNT(*) churn_rate FROM customers GROUP BY State)
   SELECT State, n, churned, ROUND(churn_rate,1) AS churn_rate_pct, RANK() OVER (ORDER BY churn_rate DESC) AS churn_rank FROM s ORDER BY churn_rank LIMIT 5;

-- q6_case_segments
SELECT CASE WHEN "Customer service calls">=4 THEN '4+ calls' WHEN "Customer service calls">=2 THEN '2-3 calls' ELSE '0-1 calls' END AS seg,
   COUNT(*) n, ROUND(100.0*AVG(Churn),1) AS churn_rate_pct FROM customers GROUP BY seg ORDER BY churn_rate_pct DESC;

-- q7_join_region
SELECT r.Region, COUNT(*) AS customers, SUM(c.Churn) AS churned, ROUND(100.0*AVG(c.Churn),1) AS churn_rate_pct,
   ROUND(AVG(c."Total day minutes"),1) AS avg_day_min FROM customers c JOIN state_region r ON c.State=r.State GROUP BY r.Region ORDER BY churn_rate_pct DESC;

-- q8_intl_plan
SELECT "International plan" AS intl_plan, COUNT(*) n, ROUND(100.0*AVG(Churn),1) AS churn_rate_pct FROM customers GROUP BY 1;

-- Optimisation
EXPLAIN QUERY PLAN SELECT * FROM customers WHERE State="TX" AND Churn=1;
CREATE INDEX idx_customers_state_churn ON customers(State, Churn);
EXPLAIN QUERY PLAN SELECT * FROM customers WHERE State="TX" AND Churn=1;
