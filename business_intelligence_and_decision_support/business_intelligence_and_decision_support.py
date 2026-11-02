ROOT = "/mnt/user-data/outputs/Codveda_BA_Internship/"   # change to your own folder; the DATA path below must point to the raw datasets
import pandas as pd, numpy as np, sqlite3, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

BASE = "/home/claude/d/Data Set For Task/Churn Prdiction Data/"
OUT = "/mnt/user-data/outputs/"

# ======================= L2 T1 : BI KPIs + dashboard preview =======================
s = pd.read_csv(ROOT + "data_preparation_and_exploration/data_cleaning/sentiment_cleaned.csv", parse_dates=["Timestamp"])
tot = len(s)
k = s.Sentiment_Group.value_counts()
print("TOTAL POSTS", tot, k.to_dict())
print("POSITIVE %%: %.1f  NEG %%: %.1f  NEU %%: %.1f" % tuple(100 * k[x] / tot for x in ["Positive", "Negative", "Neutral"]))
print("Likes", s.Likes.sum(), "Retweets", s.Retweets.sum(), "AvgEng/post", round(s.Engagement.mean(), 2))
plat = pd.crosstab(s.Platform, s.Sentiment_Group, normalize="index").round(3) * 100
print("\nSENTIMENT % BY PLATFORM\n", plat)
top = s.Country.value_counts().head(5).index
print("\nSENTIMENT % BY TOP-5 COUNTRY\n", (pd.crosstab(s.Country, s.Sentiment_Group, normalize="index").loc[top] * 100).round(1))
print("\nTOP COUNTRIES (posts)\n", s.Country.value_counts().head(5).to_dict())
tags = s.assign(tag=s.Hashtags.str.findall(r"#\w+")).explode("tag").dropna(subset=["tag"])
ht = tags.groupby("tag").agg(posts=("Text", "size"), likes=("Likes", "sum"), retweets=("Retweets", "sum")).sort_values("posts", ascending=False)
print("\nTOP HASHTAGS\n", ht.head(8))
ht.to_csv(ROOT + "business_intelligence_and_decision_support/bi_reporting/hashtag_summary.csv")
yr = s.groupby([s.Timestamp.dt.year, "Sentiment_Group"]).size().unstack(fill_value=0)
print("\nPOSTS BY YEAR x SENTIMENT\n", yr)
print("Busiest month:", s.groupby(s.Timestamp.dt.to_period("M")).size().sort_values(ascending=False).head(3).to_dict())
neg_share_yr = (yr["Negative"] / yr.sum(axis=1) * 100).round(1)
print("NEG % by year:", neg_share_yr.to_dict())

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(15, 9))
fig.suptitle("Social Media Sentiment Dashboard (preview)  |  Posts: %d  |  Positive: %.1f%%  |  Avg engagement/post: %.1f"
             % (tot, 100 * k["Positive"] / tot, s.Engagement.mean()), fontsize=14, weight="bold")
ax = fig.add_subplot(2, 3, 1); yr.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#999", "#5cb85c"]); ax.set_title("Posts by year & sentiment")
ax = fig.add_subplot(2, 3, 2); plat.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#999", "#5cb85c"], rot=0); ax.set_title("Sentiment % by platform")
ax = fig.add_subplot(2, 3, 3); k.plot(kind="pie", ax=ax, autopct="%1.0f%%", colors=["#5cb85c", "#d9534f", "#999"]); ax.set_ylabel(""); ax.set_title("Sentiment distribution")
ax = fig.add_subplot(2, 3, 4); (pd.crosstab(s.Country, s.Sentiment_Group).loc[top]).plot(kind="bar", ax=ax, rot=0, color=["#d9534f", "#999", "#5cb85c"]); ax.set_title("Top-5 countries by sentiment")
ax = fig.add_subplot(2, 3, 5); ht.head(8)[::-1].posts.plot(kind="barh", ax=ax); ax.set_title("Top hashtags (posts)")
ax = fig.add_subplot(2, 3, 6); s.groupby("Platform")[["Likes", "Retweets"]].mean().plot(kind="bar", ax=ax, rot=0); ax.set_title("Avg likes / retweets by platform")
plt.tight_layout(rect=[0, 0, 1, .95]); plt.savefig(ROOT + "Level_2/Task_1_BI_Reporting/bi_dashboard_preview.png", dpi=100); plt.close()

# ======================= L2 T2 : SQL (SQLite) =======================
a = pd.read_csv(BASE + "churn-bigml-80.csv"); b = pd.read_csv(BASE + "churn-bigml-20.csv")
a["Source_File"] = "bigml-80"; b["Source_File"] = "bigml-20"
c = pd.concat([a, b], ignore_index=True)
c.insert(0, "Customer_ID", range(1, len(c) + 1))
c["Churn"] = c["Churn"].astype(int)
REG = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
       "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX", "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
reg = pd.DataFrame([(st, r) for r, v in REG.items() for st in v.split()], columns=["State", "Region"])
db = ROOT + "business_intelligence_and_decision_support/sql_analysis/churn.db"
import os
if os.path.exists(db): os.remove(db)
con = sqlite3.connect(db)
c.to_sql("customers", con, index=False); reg.to_sql("state_region", con, index=False)
Q = {
 "q1_count": 'SELECT COUNT(*) AS total_customers FROM customers;',
 "q2_churn_dist": 'SELECT Churn, COUNT(*) AS customer_count, ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM customers),2) AS pct FROM customers GROUP BY Churn;',
 "q3_avg_by_churn": '''SELECT Churn, COUNT(*) AS n, ROUND(AVG("Total day minutes"),2) AS avg_day_min,
   ROUND(AVG("Customer service calls"),2) AS avg_service_calls,
   ROUND(100.0*SUM("International plan"='Yes')/COUNT(*),1) AS pct_intl_plan,
   ROUND(100.0*SUM("Voice mail plan"='Yes')/COUNT(*),1) AS pct_vmail_plan FROM customers GROUP BY Churn;''',
 "q4_state_service_calls": '''SELECT State, COUNT(*) AS customers, ROUND(AVG("Customer service calls"),2) AS avg_service_calls
   FROM customers GROUP BY State HAVING COUNT(*)>=10 ORDER BY avg_service_calls DESC LIMIT 5;''',
 "q5_cte_state_churn_rank": '''WITH s AS (SELECT State, COUNT(*) n, SUM(Churn) churned, 100.0*SUM(Churn)/COUNT(*) churn_rate FROM customers GROUP BY State)
   SELECT State, n, churned, ROUND(churn_rate,1) AS churn_rate_pct, RANK() OVER (ORDER BY churn_rate DESC) AS churn_rank FROM s ORDER BY churn_rank LIMIT 5;''',
 "q6_case_segments": '''SELECT CASE WHEN "Customer service calls">=4 THEN '4+ calls' WHEN "Customer service calls">=2 THEN '2-3 calls' ELSE '0-1 calls' END AS seg,
   COUNT(*) n, ROUND(100.0*AVG(Churn),1) AS churn_rate_pct FROM customers GROUP BY seg ORDER BY churn_rate_pct DESC;''',
 "q7_join_region": '''SELECT r.Region, COUNT(*) AS customers, SUM(c.Churn) AS churned, ROUND(100.0*AVG(c.Churn),1) AS churn_rate_pct,
   ROUND(AVG(c."Total day minutes"),1) AS avg_day_min FROM customers c JOIN state_region r ON c.State=r.State GROUP BY r.Region ORDER BY churn_rate_pct DESC;''',
 "q8_intl_plan": '''SELECT "International plan" AS intl_plan, COUNT(*) n, ROUND(100.0*AVG(Churn),1) AS churn_rate_pct FROM customers GROUP BY 1;''',
}
res = {}
for n, q in Q.items():
    res[n] = pd.read_sql(q, con); print("\n--", n, "\n", res[n].to_string(index=False))
# optimisation evidence
plan_q = 'SELECT * FROM customers WHERE State="TX" AND Churn=1;'
before = con.execute("EXPLAIN QUERY PLAN " + plan_q).fetchall()
con.execute('CREATE INDEX idx_customers_state_churn ON customers(State, Churn);')
after = con.execute("EXPLAIN QUERY PLAN " + plan_q).fetchall()
print("\nPLAN BEFORE:", before[0][3], "\nPLAN AFTER :", after[0][3])
con.commit(); con.close()
with open(ROOT + "Level_2/Task_2_SQL/churn_queries.sql", "w") as f:
    f.write("-- Schema: customers(Customer_ID, State, ..., Churn, Source_File) ; state_region(State, Region)\n\n")
    for n, q in Q.items(): f.write(f"-- {n}\n{q}\n\n")
    f.write("-- Optimisation\nEXPLAIN QUERY PLAN " + plan_q + "\nCREATE INDEX idx_customers_state_churn ON customers(State, Churn);\nEXPLAIN QUERY PLAN " + plan_q + "\n")

# ======================= L2 T3 : STATISTICS =======================
c["Churn"] = c["Churn"].astype(bool)
ch, nc = c[c.Churn]["Customer service calls"], c[~c.Churn]["Customer service calls"]
print("\n=== WELCH T-TEST service calls ===")
print("n churn/no churn:", len(ch), len(nc), "means: %.3f vs %.3f diff %.3f" % (ch.mean(), nc.mean(), ch.mean() - nc.mean()))
t = stats.ttest_ind(ch, nc, equal_var=False)
v1, v2 = ch.var(ddof=1) / len(ch), nc.var(ddof=1) / len(nc)
dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(ch) - 1) + v2 ** 2 / (len(nc) - 1))
se = np.sqrt(v1 + v2); d = ch.mean() - nc.mean()
print("t=%.3f df=%.1f p=%.2e  95%% CI diff [%.3f, %.3f]" % (t.statistic, dof, t.pvalue, d - stats.t.ppf(.975, dof) * se, d + stats.t.ppf(.975, dof) * se))
sp = np.sqrt((ch.var() + nc.var()) / 2); print("Cohen d = %.3f" % (d / sp))
print("Mann-Whitney p = %.2e" % stats.mannwhitneyu(ch, nc).pvalue)

print("\n=== CHI-SQUARE intl plan vs churn ===")
ct = pd.crosstab(c["International plan"], c["Churn"]); print(ct)
chi, p, dof2, exp = stats.chi2_contingency(ct, correction=False)
print("chi2=%.2f df=%d p=%.2e min expected=%.1f" % (chi, dof2, p, exp.min()))
print("churn rate Yes/No:", (ct.loc["Yes", True] / ct.loc["Yes"].sum()).round(4), (ct.loc["No", True] / ct.loc["No"].sum()).round(4))
print("Cramer V = %.3f" % np.sqrt(chi / len(c)))
ct2 = pd.crosstab(c["Voice mail plan"], c["Churn"]); chi2b, p2b, _, _ = stats.chi2_contingency(ct2, correction=False)
print("VoiceMail plan chi2=%.2f p=%.2e" % (chi2b, p2b), ct2.to_dict())

print("\n=== DISTRIBUTIONS ===")
sc = c["Customer service calls"]; print("service calls mean %.3f var %.3f (Poisson needs mean≈var)" % (sc.mean(), sc.var()))
obs = sc.value_counts().sort_index(); lam = sc.mean()
exp_p = pd.Series({i: len(sc) * stats.poisson.pmf(i, lam) for i in range(0, 6)}); print(pd.DataFrame({"obs": obs.reindex(range(6)), "poisson_exp": exp_p.round(1)}))
dm = c["Total day minutes"]; print("day minutes skew %.3f kurt %.3f  Shapiro(n=5000 cap) p=%.3f" % (dm.skew(), dm.kurt(), stats.shapiro(dm).pvalue))
print("P(service calls>=4) observed %.3f ; Poisson predicts %.3f" % ((sc >= 4).mean(), 1 - stats.poisson.cdf(3, lam)))

print("\n=== CHURN PROPORTION CI ===")
x, n = int(c.Churn.sum()), len(c); ph = x / n
moe = 1.96 * np.sqrt(ph * (1 - ph) / n)
print("p=%.4f (%d/%d)  normal 95%% CI [%.4f, %.4f] MOE=%.4f" % (ph, x, n, ph - moe, ph + moe, moe))
print("Wilson 95%% CI", np.round(proportion_confint(x, n, method="wilson"), 4))
print("99%% Wilson CI", np.round(proportion_confint(x, n, alpha=.01, method="wilson"), 4))
print("Mean service calls 95%% CI", np.round(stats.t.interval(.95, n - 1, sc.mean(), stats.sem(sc)), 3))

print("\n=== TWO-PROPORTION TEST (OBSERVATIONAL, not A/B) voicemail plan vs churn ===")
g = c.groupby("Voice mail plan").Churn.agg(["sum", "count"]); print(g)
z, pz = proportions_ztest(g["sum"].values, g["count"].values)
lo, hi = confint_proportions_2indep(g.loc["Yes", "sum"], g.loc["Yes", "count"], g.loc["No", "sum"], g.loc["No", "count"], method="wald")
print("z=%.3f p=%.2e rate diff Yes-No CI [%.4f, %.4f]" % (z, pz, lo, hi))

print("\n=== A/B TEST DESIGN (sample size) ===")
for base, tgt in [(0.145, 0.12), (0.145, 0.10)]:
    es = proportion_effectsize(base, tgt)
    nn = NormalIndPower().solve_power(es, alpha=.05, power=.8, alternative="two-sided")
    print("baseline churn %.1f%% -> %.1f%% : %d customers per group (80%% power, alpha .05)" % (base * 100, tgt * 100, np.ceil(nn)))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
sns.countplot(data=c, x="Customer service calls", hue="Churn", ax=ax[0]); ax[0].set_title("Service calls by churn")
sns.barplot(data=c, x="Customer service calls", y="Churn", ax=ax[1], color="#d9534f"); ax[1].set_title("Churn rate by # service calls")
ax[2].bar(obs.index - .2, obs.values, .4, label="observed"); ax[2].bar(exp_p.index + .2, exp_p.values, .4, label="Poisson"); ax[2].legend(); ax[2].set_title("Service calls: observed vs Poisson")
plt.tight_layout(); plt.savefig(ROOT + "Level_2/Task_3_Statistics/stats_charts.png", dpi=105)
