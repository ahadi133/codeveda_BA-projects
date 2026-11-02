ROOT = "/mnt/user-data/outputs/Codveda_BA_Internship/"   # change to your own folder; the DATA path below must point to the raw datasets
import pandas as pd, numpy as np, joblib, pulp, warnings, matplotlib
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

D = "/home/claude/d/Data Set For Task/Churn Prdiction Data/"; OUT = "/mnt/user-data/outputs/"
SEED = 42
te = pd.read_csv(D + "churn-bigml-20.csv")
pr = pd.read_csv(ROOT + "predictive_risk_and_strategy_analytics/predictive_modeling/l3_churn_test_probs.csv")
te["Customer_ID"] = ["C%04d" % i for i in range(1, len(te) + 1)]
te["p_churn"] = pr.p_churn.values; te["actual"] = pr.actual.values

# =================== TASK 2 : RISK ASSESSMENT (Option A: churn risk) ===================
bins = [-.001, .30, .60, 1.001]; labels = ["Low", "Medium", "High"]
te["risk_band"] = pd.cut(te.p_churn, bins=bins, labels=labels)
rb = te.groupby("risk_band", observed=True).agg(customers=("actual", "size"), predicted_avg=("p_churn", "mean"), actual_churn_rate=("actual", "mean"), churners=("actual", "sum")).round(3)
rb["share_of_all_churners"] = (rb.churners / te.actual.sum()).round(3); print("RISK BANDS (illustrative 30/60 thresholds)\n", rb.to_string())
print("Brier score %.4f (always-base-rate: %.4f), AUC %.3f" % (brier_score_loss(te.actual, te.p_churn), brier_score_loss(te.actual, np.repeat(te.actual.mean(), len(te))), roc_auc_score(te.actual, te.p_churn)))
frac, mean_p = calibration_curve(te.actual, te.p_churn, n_bins=6, strategy="quantile"); print("calibration (pred -> actual):", list(zip(mean_p.round(2), frac.round(2))))
# group churn rates (exploratory risk analysis, full test file)
print("\nGROUP CHURN RATES (20% file)")
print(te.groupby("International plan").actual.mean().round(3).to_dict(), te.groupby("Voice mail plan").actual.mean().round(3).to_dict())
print(te.groupby(pd.cut(te["Customer service calls"], [-1, 1, 3, 20], labels=["0-1", "2-3", "4+"]), observed=True).actual.agg(["mean", "size"]).round(3).to_dict("index"))
te[["Customer_ID", "State", "International plan", "Voice mail plan", "Customer service calls", "p_churn", "risk_band", "actual"]].sort_values("p_churn", ascending=False).to_csv(ROOT + "Level_3/Task_2_Risk_Analysis/l3_risk_scores.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(mean_p, frac, "o-"); ax[0].plot([0, 1], [0, 1], "--", c="grey"); ax[0].set_title("Calibration (20% test file)"); ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual")
rb.customers.plot(kind="bar", ax=ax[1], rot=0, color=["#5cb85c", "#f0ad4e", "#d9534f"]); ax[1].set_title("Customers per risk band")
plt.tight_layout(); plt.savefig(ROOT + "Level_3/Task_2_Risk_Analysis/l3_risk_charts.png", dpi=105); plt.close()

# =================== TASK 3 : PRESCRIPTIVE (all inputs below are ASSUMPTIONS) ===================
A = dict(margin_rate=0.30, horizon_months=12, offer_discount=0.10, offer_months=3, admin_fee=5.0, success=0.30, budget=1500.0,
         broad_cost_factor=0.5, broad_success=0.15)
te["monthly_bill"] = te[[c for c in te.columns if "charge" in c]].sum(axis=1)
te["value"] = te.monthly_bill * A["horizon_months"] * A["margin_rate"]              # value of retaining customer (12-mo margin)
te["cost"] = te.monthly_bill * A["offer_discount"] * A["offer_months"] + A["admin_fee"]   # cost of offer
def benefit(d, success=A["success"], cost_mult=1.0, val_mult=1.0):
    return d.p_churn * success * d.value * val_mult - d.cost * cost_mult                # NET expected benefit (cost already subtracted)
te["net_benefit"] = benefit(te)
print("\nASSUMPTIONS", A); print("avg value $%.0f  avg offer cost $%.1f  cost of offering to everyone $%.0f  customers with net_benefit>0: %d / %d" %
      (te.value.mean(), te.cost.mean(), te.cost.sum(), (te.net_benefit > 0).sum(), len(te)))

def solve(d, budget, success=A["success"], cost_mult=1.0, val_mult=1.0):
    b = benefit(d, success, cost_mult, val_mult).values; c = d.cost.values * cost_mult
    m = pulp.LpProblem("retention", pulp.LpMaximize); x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(d))]
    m += pulp.lpSum(b[i] * x[i] for i in range(len(d))); m += pulp.lpSum(c[i] * x[i] for i in range(len(d))) <= budget
    m.solve(pulp.PULP_CBC_CMD(msg=0)); sel = np.array([int(v.value() > .5) for v in x]); return sel, b, c

# greedy strategies
def pick(order, cost, budget):
    sel = np.zeros(len(order), int); spent = 0
    for i in order:
        if spent + cost[i] <= budget: sel[i] = 1; spent += cost[i]
    return sel
C0 = te.cost.values
sel_opt, b0, _ = solve(te, A["budget"])
sel_risk = pick(np.argsort(-te.p_churn.values), C0, A["budget"])
sel_broad = pick(np.argsort(-te.p_churn.values), C0 * A["broad_cost_factor"], A["budget"])   # half-price offers to twice as many
strategies = {"A. No campaign": (np.zeros(len(te), int), 0.0, 0.0),
              "B. Highest-risk first (full offer)": (sel_risk, 1.0, A["success"]),
              "C. Optimised (max expected net benefit)": (sel_opt, 1.0, A["success"]),
              "D. Broad smaller incentive (half cost)": (sel_broad, A["broad_cost_factor"], A["broad_success"])}
rng = np.random.default_rng(SEED); N = 5000; p = te.p_churn.values; V = te.value.values
rows = []
for n, (sel, cm, sc) in strategies.items():
    cost = (C0 * cm * sel).sum()
    churn_draw = rng.random((N, len(te))) < p                                    # who would churn without help
    save_draw = rng.random((N, len(te))) < sc                                    # who is saved if offered
    saved = churn_draw & save_draw & (sel == 1)
    retained = saved.sum(1); gain = (saved * V).sum(1) - cost
    exp_gain = (p * sc * V * sel).sum() - cost
    rows.append(dict(strategy=n, customers_offered=int(sel.sum()), campaign_cost=round(cost, 0), expected_retained=round(retained.mean(), 1),
                     expected_net_benefit=round(gain.mean(), 0), p5=round(np.percentile(gain, 5), 0), p95=round(np.percentile(gain, 95), 0), analytic_check=round(exp_gain, 0),
                     prob_loss=round((gain < 0).mean(), 3)))
sim = pd.DataFrame(rows); print("\nSIMULATION (%d runs, seed %d)\n" % (N, SEED), sim.to_string(index=False)); sim.to_csv(ROOT + "Level_3/Task_3_Prescriptive/l3_strategy_simulation.csv", index=False)
print("optimal: selected %d customers, cost $%.0f, expected net $%.0f; LP check vs greedy-by-ratio" % (sel_opt.sum(), (C0 * sel_opt).sum(), (b0 * sel_opt).sum()))
print("overlap of optimised vs highest-risk selections: %d customers" % ((sel_opt & sel_risk).sum()))

# scenario analysis
sc_rows = []
base_sel = sel_opt
scen = {"Base": dict(budget=A["budget"]), "Budget -20%": dict(budget=A["budget"] * .8), "Offer cost +25%": dict(budget=A["budget"], cost_mult=1.25),
        "Success rate 20% (lower)": dict(budget=A["budget"], success=.20), "Success rate 40%": dict(budget=A["budget"], success=.40),
        "Customer value -30%": dict(budget=A["budget"], val_mult=.7), "Customer value +30%": dict(budget=A["budget"], val_mult=1.3),
        "Success 10%": dict(budget=A["budget"], success=.10)}
for n, kw in scen.items():
    sel, b, c = solve(te, **kw)
    sc_rows.append(dict(scenario=n, budget=round(kw["budget"]), offered=int(sel.sum()), cost=round((c * sel).sum()), expected_net_benefit=round((b * sel).sum()),
                        overlap_with_base=int((sel & base_sel).sum())))
scn = pd.DataFrame(sc_rows); print("\nSCENARIOS\n", scn.to_string(index=False)); scn.to_csv(ROOT + "Level_3/Task_3_Prescriptive/l3_scenario_comparison.csv", index=False)

# ranked recommendations (human review required)
rec = te.assign(selected=sel_opt, expected_net_benefit=b0).query("selected==1").sort_values("expected_net_benefit", ascending=False)
rec["recommended_action"] = np.where(rec["Customer service calls"] >= 4, "Offer + support-team callback", "Offer retention discount")
rec[["Customer_ID", "p_churn", "risk_band", "cost", "value", "expected_net_benefit", "recommended_action"]].round(2).to_csv(ROOT + "Level_3/Task_3_Prescriptive/l3_recommended_allocation.csv", index=False)
print("\nTOP 5 RECOMMENDATIONS\n", rec[["Customer_ID", "p_churn", "cost", "value", "expected_net_benefit", "recommended_action"]].head(5).round(2).to_string(index=False))
print("selected by band:", rec.risk_band.value_counts().to_dict(), "| avg p_churn selected %.2f" % rec.p_churn.mean())
pd.DataFrame([(k, v) for k, v in A.items()], columns=["assumption", "value"]).to_csv(ROOT + "Level_3/Task_3_Prescriptive/l3_assumptions_register.csv", index=False)
