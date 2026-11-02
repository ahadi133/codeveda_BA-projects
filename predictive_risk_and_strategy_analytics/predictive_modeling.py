ROOT = "/mnt/user-data/outputs/Codveda_BA_Internship/"   # change to your own folder; the DATA path below must point to the raw datasets
import pandas as pd, numpy as np, joblib, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score, RocCurveDisplay)
from sklearn.cluster import KMeans
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

D = "/home/claude/d/Data Set For Task/"; OUT = "/mnt/user-data/outputs/"
SEED = 42

# ---------------- 1. CHURN CLASSIFICATION ----------------
tr = pd.read_csv(D + "Churn Prdiction Data/churn-bigml-80.csv"); te = pd.read_csv(D + "Churn Prdiction Data/churn-bigml-20.csv")
print("missing", tr.isna().sum().sum(), te.isna().sum().sum(), "dups", tr.duplicated().sum(), "churn rate tr/te", tr.Churn.mean().round(4), te.Churn.mean().round(4))
CHARGE = [c for c in tr.columns if "charge" in c]       # charge = minutes x fixed rate -> perfectly collinear
def prep(d, drop_charge=False):
    X = d.drop(columns=["Churn"]).copy()
    X["International plan"] = (X["International plan"] == "Yes").astype(int)
    X["Voice mail plan"] = (X["Voice mail plan"] == "Yes").astype(int)
    if drop_charge: X = X.drop(columns=CHARGE)
    return X
print("corr(day minutes, day charge) =", tr["Total day minutes"].corr(tr["Total day charge"]).round(4))
# State handled by one-hot; Area code treated as categorical
def build(d, cols=None):
    X = prep(d, drop_charge=True); X = pd.get_dummies(X, columns=["State", "Area code"], dtype=int)
    if cols is not None: X = X.reindex(columns=cols, fill_value=0)
    return X
Xtr = build(tr); Xte = build(te, Xtr.columns); ytr, yte = tr.Churn.astype(int), te.Churn.astype(int)
Xtr.assign(Churn=ytr).to_csv(ROOT + "predictive_risk_and_strategy_analytics/predictive_modeling/churn_model_ready_train.csv", index=False)

models = {
 "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
 "Logistic Reg (balanced)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
 "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5, random_state=SEED),
 "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
}
rows = []; fitted = {}
for n, m in models.items():
    m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yh = (p >= .5).astype(int); fitted[n] = (m, p)
    cm = confusion_matrix(yte, yh)
    rows.append(dict(model=n, accuracy=accuracy_score(yte, yh), precision=precision_score(yte, yh), recall=recall_score(yte, yh),
                     f1=f1_score(yte, yh), roc_auc=roc_auc_score(yte, p), TN=cm[0, 0], FP=cm[0, 1], FN=cm[1, 0], TP=cm[1, 1]))
res = pd.DataFrame(rows).round(3); print("\nCHURN MODELS (test = 20% file, threshold .5)\n", res.to_string(index=False)); res.to_csv(ROOT + "Level_3/Task_1_Predictive_ML/l3_churn_model_comparison.csv", index=False)
print("baseline accuracy (always 'no churn'):", round(1 - yte.mean(), 3))
cv = cross_val_score(models["Random Forest"], Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="roc_auc")
print("RF 5-fold CV AUC on train: %.3f +/- %.3f" % (cv.mean(), cv.std()))
rf = fitted["Random Forest"][0]
imp = pd.Series(rf.feature_importances_, Xtr.columns).sort_values(ascending=False).head(8).round(3); print("RF top features\n", imp)
# threshold sweep for chosen model
p = fitted["Random Forest"][1]
for t in [.2, .3, .4, .5]:
    yh = (p >= t).astype(int); print("RF thr %.1f precision %.3f recall %.3f f1 %.3f flagged %d" % (t, precision_score(yte, yh), recall_score(yte, yh), f1_score(yte, yh), yh.sum()))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for n, (m, pp) in fitted.items(): RocCurveDisplay.from_predictions(yte, pp, name=n, ax=ax[0])
sns.heatmap(confusion_matrix(yte, (p >= .5).astype(int)), annot=True, fmt="d", cmap="Blues", ax=ax[1]); ax[1].set_title("RF confusion matrix (thr .5)")
plt.tight_layout(); plt.savefig(ROOT + "Level_3/Task_1_Predictive_ML/l3_churn_roc_cm.png", dpi=105); plt.close()
# final deployable model: RF on train+test with raw-input pipeline
joblib.dump({"model": rf, "columns": list(Xtr.columns)}, ROOT + "Level_3/Task_1_Predictive_ML/churn_model.joblib")
pd.DataFrame({"actual": yte.values, "p_churn": p}).to_csv(ROOT + "Level_3/Task_1_Predictive_ML/l3_churn_test_probs.csv", index=False)

# ---------------- 2. HOUSE REGRESSION ----------------
cols = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
h = pd.read_csv(D + "4) house Prediction Data Set.csv", sep=r"\s+", header=None, names=cols)
print("\nHOUSE", h.shape, "missing", h.isna().sum().sum(), "MEDV mean %.2f cap-count(=50): %d" % (h.MEDV.mean(), (h.MEDV == 50).sum()))
h.to_csv(ROOT + "Level_3/Task_1_Predictive_ML/house_cleaned.csv", index=False)
feat = [c for c in cols if c != "MEDV"]
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(h[feat], h.MEDV, test_size=.2, random_state=SEED)
reg = {"Linear Regression": LinearRegression(), "Random Forest Regressor": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)}
rr = []; hp = {}
for n, m in reg.items():
    m.fit(Xh_tr, yh_tr); pr = m.predict(Xh_te); hp[n] = pr
    rr.append(dict(model=n, MAE=mean_absolute_error(yh_te, pr), RMSE=np.sqrt(mean_squared_error(yh_te, pr)), R2=r2_score(yh_te, pr)))
rr = pd.DataFrame(rr).round(3); print(rr.to_string(index=False)); rr.to_csv(ROOT + "Level_3/Task_1_Predictive_ML/l3_house_model_comparison.csv", index=False)
print("LinReg strongest coefs (per unit):", pd.Series(reg["Linear Regression"].coef_, feat).round(2).sort_values().iloc[[0, 1, -2, -1]].to_dict())
print("RF top features:", pd.Series(reg["Random Forest Regressor"].feature_importances_, feat).sort_values(ascending=False).head(3).round(3).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, (n, pr) in zip(ax, hp.items()):
    a.scatter(yh_te, pr, s=14); a.plot([0, 50], [0, 50], "r--"); a.set_title(n); a.set_xlabel("Actual MEDV ($1000s)"); a.set_ylabel("Predicted")
plt.tight_layout(); plt.savefig(ROOT + "Level_3/Task_1_Predictive_ML/l3_house_pred_vs_actual.png", dpi=105); plt.close()
joblib.dump({"model": reg["Random Forest Regressor"], "columns": feat}, ROOT + "Level_3/Task_1_Predictive_ML/house_model.joblib")

# ---------------- 3. CLUSTERING (usage-based segments) ----------------
allc = pd.concat([tr, te], ignore_index=True)
allc["Total charge"] = allc[CHARGE].sum(axis=1)
allc["Total calls"] = allc[[c for c in allc.columns if c.startswith("Total") and c.endswith("calls")]].sum(axis=1)
seg_f = ["Total day minutes", "Total intl calls", "Customer service calls", "Total charge"]
Z = StandardScaler().fit_transform(allc[seg_f])
sil = {k: round(silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)), 3) for k in range(2, 8)}
print("\nSILHOUETTE by k:", sil)
K = 4
allc["Segment"] = KMeans(K, n_init=10, random_state=SEED).fit_predict(Z)
prof = allc.groupby("Segment").agg(n=("Churn", "size"), churn_rate=("Churn", "mean"), day_min=("Total day minutes", "mean"), intl_calls=("Total intl calls", "mean"),
                                   svc_calls=("Customer service calls", "mean"), total_charge=("Total charge", "mean")).round(2)
print("PROFILE (k=%d)\n" % K, prof)
prof.to_csv(ROOT + "Level_3/Task_1_Predictive_ML/l3_cluster_profile.csv")
sil_best = max(sil, key=sil.get); print("best-silhouette k =", sil_best)
Zb = KMeans(sil_best, n_init=10, random_state=SEED).fit_predict(Z); allc["Seg_best"] = Zb
print(allc.groupby("Seg_best").agg(n=("Churn", "size"), churn=("Churn", "mean"), day_min=("Total day minutes", "mean"), svc=("Customer service calls", "mean"), chg=("Total charge", "mean")).round(2))

# ---------------- 4. FORECASTING (sentiment post volume, NOT sales) ----------------
s = pd.read_csv(ROOT + "data_preparation_and_exploration/data_cleaning/sentiment_cleaned.csv", parse_dates=["Timestamp"])
m = s.set_index("Timestamp").resample("MS").size(); m = m["2015-01":]
trn, tst = m[:"2023-04"], m["2023-05":]
fc = {"Naive (last value)": np.repeat(trn.iloc[-1], len(tst)), "Mean of last 6 months": np.repeat(trn.iloc[-6:].mean(), len(tst)),
      "Mean of training": np.repeat(trn.mean(), len(tst)),
      "Simple Exp. Smoothing": SimpleExpSmoothing(trn.values.astype(float), initialization_method="estimated").fit().forecast(len(tst)),
      "Holt (trend)": ExponentialSmoothing(trn.values.astype(float), trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(len(tst))}
fr = pd.DataFrame([dict(model=k, MAE=mean_absolute_error(tst, v), RMSE=np.sqrt(mean_squared_error(tst, v))) for k, v in fc.items()]).round(2)
print("\nFORECAST monthly post volume; train %s..%s (%d m), test %s..%s (%d m, actual mean %.1f)" % (trn.index[0].date(), trn.index[-1].date(), len(trn), tst.index[0].date(), tst.index[-1].date(), len(tst), tst.mean()))
print(fr.to_string(index=False)); fr.to_csv(ROOT + "Level_3/Task_1_Predictive_ML/l3_forecast_errors.csv", index=False)
plt.figure(figsize=(10, 4)); plt.plot(m.index, m.values, label="actual")
for k in ["Naive (last value)", "Mean of last 6 months"]: plt.plot(tst.index, fc[k], "--", label=k)
plt.axvline(tst.index[0], color="grey"); plt.legend(); plt.title("Monthly post volume forecast (sentiment data)"); plt.tight_layout(); plt.savefig(ROOT + "Level_3/Task_1_Predictive_ML/l3_forecast.png", dpi=105)
