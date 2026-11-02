ROOT = "/mnt/user-data/outputs/Codveda_BA_Internship/"   # change to your own folder; the DATA path below must point to the raw datasets
import pandas as pd, numpy as np, seaborn as sns, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE = "/home/claude/d/Data Set For Task/"
OUT = "/mnt/user-data/outputs/"

# =============== L1 TASK 2 : DATA COLLECTION & CLEANING (Sentiment) ===============
raw = pd.read_csv(BASE + "3) Sentiment dataset.csv")
log = {"rows_raw": len(raw), "cols_raw": raw.shape[1]}
df = raw.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])          # junk index columns
log["missing_values_total"] = int(df.isna().sum().sum())
log["exact_dups_with_index_cols"] = int(raw.duplicated().sum())

# standardise text fields (leading/trailing/padded spaces, case)
for c in ["Text", "Sentiment", "Platform", "Hashtags", "Country", "User"]:
    df[c] = df[c].astype(str).str.strip()
df["Sentiment_Detail"] = df["Sentiment"].str.title()
df["Platform"] = df["Platform"].str.title()
df["Hashtags"] = df["Hashtags"].replace("", "No Hashtag")

# group 191 emotion labels into Positive / Negative / Neutral (documented judgement mapping)
NEG = """Despair Grief Loneliness Sad Embarrassed Frustration Regret Melancholy Hate Bad Disgust Bitterness Frustrated
Betrayal Negative Boredom Overwhelmed Desolation Bitter Shame Jealousy Jealous Resentment Fearful Devastated Envious
Dismissive Heartbreak Anger Fear Sadness Disappointed Anxiety Intimidation Helplessness Envy Apprehensive Isolation
Disappointment Sorrow Loss Suffering Emotionalstorm Lostlove Exhaustion Darkness Desperation Ruins Heartache Solitude
Obstacle Pressure Miscalculation""".split()
NEU = """Neutral Curiosity Indifference Ambivalence Acceptance Numbness Reflection Contemplation Pensive Emotion Suspense
Bittersweet Confusion Arousal Intrigue Anticipation Surprise Exploration Journey Mindfulness Immersion Yearning
Challenge Engagement Nostalgia Iconic""".split()
def group(s):
    k = s.replace(" ", "")
    if k in [n.replace(" ", "") for n in NEG]: return "Negative"
    if k in [n.replace(" ", "") for n in NEU]: return "Neutral"
    return "Positive"
df["Sentiment_Group"] = df["Sentiment_Detail"].map(group)

# types
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Retweets"] = pd.to_numeric(df["Retweets"], errors="coerce")
df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce")
log["invalid_dates"] = int(df.Timestamp.isna().sum())
log["negative_counts"] = int(((df.Likes < 0) | (df.Retweets < 0)).sum())

# duplicates (after standardising)
dups = df.duplicated(keep="first")
log["dups_removed_after_standardising"] = int(dups.sum())
df = df[~dups].reset_index(drop=True)

# outliers (IQR) - flagged, NOT removed
def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
log["likes_outliers_flagged"] = iqr_out(df.Likes)
log["retweets_outliers_flagged"] = iqr_out(df.Retweets)
df["Engagement"] = df.Likes + df.Retweets
df["Year"] = df.Timestamp.dt.year; df["Month"] = df.Timestamp.dt.month
df["Day"] = df.Timestamp.dt.day; df["Hour"] = df.Timestamp.dt.hour
df.to_csv(ROOT + "data_preparation_and_exploration/data_cleaning/sentiment_cleaned.csv", index=False)
log["rows_clean"] = len(df)
print("CLEANING LOG:", log)
print(df.Sentiment_Group.value_counts(), df.Sentiment_Group.value_counts(normalize=True).round(3))

# Iris quality checks
iris = pd.read_csv(BASE + "1) iris.csv")
print("IRIS missing:", iris.isna().sum().sum(), "dups:", iris.duplicated().sum(),
      "species:", iris.species.unique())
iris_clean = iris.drop_duplicates().reset_index(drop=True)
iris_clean.to_csv(ROOT + "data_preparation_and_exploration/data_cleaning/iris_cleaned.csv", index=False)
print("IRIS rows after dedup:", len(iris_clean))

# =============== L1 TASK 3 : EDA (Iris, original 150 rows) ===============
num = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
d = iris[num]
summ = pd.DataFrame({"mean": d.mean(), "median": d.median(), "mode": d.mode().iloc[0], "min": d.min(),
                     "max": d.max(), "range": d.max() - d.min(), "variance": d.var(), "std": d.std()}).round(3)
print("\nDESCRIPTIVES\n", summ)
summ.to_csv(ROOT + "data_preparation_and_exploration/exploratory_data_analysis/iris_descriptive_stats.csv")
print("\nBY SPECIES (mean)\n", iris.groupby("species")[num].mean().round(3))
print("\nBY SPECIES (std)\n", iris.groupby("species")[num].std().round(3))
corr = d.corr()
print("\nCORR\n", corr.round(3))
# outliers by IQR
for c in num:
    q1, q3 = d[c].quantile([.25, .75]); i = q3 - q1
    o = iris[(d[c] < q1 - 1.5 * i) | (d[c] > q3 + 1.5 * i)]
    print(c, "IQR outliers:", len(o), o[[c, "species"]].values.tolist())
# within-species correlation (shows Simpson-style caveat)
print("\nCORR petal_length~petal_width within species")
print(iris.groupby("species").apply(lambda g: g.petal_length.corr(g.petal_width), include_groups=False).round(3))
# simple regression
lr = stats.linregress(iris.petal_width, iris.petal_length)
print("\nREGRESSION petal_length ~ petal_width: slope %.3f intercept %.3f r2 %.3f p %.2e" %
      (lr.slope, lr.intercept, lr.rvalue ** 2, lr.pvalue))
# ANOVA per feature
for c in num:
    f, p = stats.f_oneway(*[g[c] for _, g in iris.groupby("species")])
    print("ANOVA", c, "F=%.1f p=%.1e" % (f, p))

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a, c in zip(ax.ravel(), num): sns.histplot(data=iris, x=c, hue="species", kde=True, ax=a)
plt.tight_layout(); plt.savefig(ROOT + "Level_1/Task_3_EDA/iris_histograms.png", dpi=110); plt.close()
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for a, c in zip(ax.ravel(), num): sns.boxplot(data=iris, x="species", y=c, hue="species", ax=a, legend=False)
plt.tight_layout(); plt.savefig(ROOT + "Level_1/Task_3_EDA/iris_boxplots.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
sns.scatterplot(data=iris, x="petal_length", y="petal_width", hue="species", ax=ax[0])
sns.scatterplot(data=iris, x="sepal_length", y="sepal_width", hue="species", ax=ax[1])
sns.regplot(data=iris, x="petal_width", y="petal_length", ax=ax[2], scatter_kws={"s": 15})
ax[2].set_title("Linear fit: petal_length ~ petal_width")
plt.tight_layout(); plt.savefig(ROOT + "Level_1/Task_3_EDA/iris_scatter.png", dpi=110); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0])
sns.countplot(data=iris, x="species", hue="species", ax=ax[1])
plt.tight_layout(); plt.savefig(ROOT + "Level_1/Task_3_EDA/iris_corr_and_counts.png", dpi=110); plt.close()

# sentiment time view (alt. EDA)
m = df.groupby(df.Timestamp.dt.to_period("Y")).size()
print("\nPosts per year\n", m)
print(df.groupby("Platform").Engagement.mean().round(1))
