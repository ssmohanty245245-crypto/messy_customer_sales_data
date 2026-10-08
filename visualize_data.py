import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw", "messy_customer_sales_data.csv")
CLEAN = os.path.join(BASE, "data", "cleaned", "cleaned_customer_sales_data.csv")
OUT = os.path.join(BASE, "charts")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", font="DejaVu Sans")
PAL = ["#2F5D8A", "#E07A5F", "#81B29A", "#F2CC8F", "#6D597A", "#B56576"]
inr = FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
raw_age = pd.to_numeric(raw["Age"].str.replace(r"\s*years", "", regex=True).replace("", np.nan), errors="coerce")
raw_amt = pd.to_numeric(raw["Purchase_Amount"].replace("", np.nan), errors="coerce")

df = pd.read_csv(CLEAN, dtype={"Phone_Number": str}, parse_dates=["Signup_Date", "Last_Purchase_Date"])
known = df[df["Customer_ID_Missing"] == False]
real_amt = df[~df["Purchase_Amount_Imputed"]]      # exclude imputed values from amount analysis
real_age = df[~df["Age_Imputed"]]

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# 1 ── Missing values: before vs after cleaning
cols = ["Customer_ID", "Gender", "Age", "City", "Last_Purchase_Date", "Purchase_Amount", "Feedback_Score", "Country"]
before = [(raw[c].str.strip() == "").sum() for c in cols]
after = [df[c].isna().sum() for c in cols]
after[0] = int(df["Customer_ID_Missing"].sum()) * 0        # surrogate IDs assigned
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cols)); w = 0.38
ax.bar(x - w/2, before, w, label="Before cleaning", color=PAL[1])
ax.bar(x + w/2, after, w, label="After cleaning", color=PAL[0])
ax.set_xticks(x); ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_ylabel("Missing values"); ax.set_title("Missing Values: Before vs After Cleaning", fontweight="bold")
ax.legend()
save(fig, "01_missing_values_before_after")

# 2 ── Outliers: before vs after (Age and Purchase Amount)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot([raw_age.dropna(), real_age["Age"]], tick_labels=["Raw", "Cleaned"], patch_artist=True,
                boxprops=dict(facecolor="#CFE0EE"), medianprops=dict(color=PAL[1], lw=2))
axes[0].set_title("Age: outliers (-10, 3, 250) removed", fontweight="bold"); axes[0].set_ylabel("Age")
axes[1].boxplot([raw_amt.dropna(), real_amt["Purchase_Amount"]], tick_labels=["Raw", "Cleaned"], patch_artist=True,
                boxprops=dict(facecolor="#CFE0EE"), medianprops=dict(color=PAL[1], lw=2))
axes[1].set_yscale("symlog", linthresh=1000)
axes[1].set_title("Purchase Amount: -500 / 0 / 9,999,999 removed (log scale)", fontweight="bold")
axes[1].set_ylabel("Amount")
save(fig, "02_outliers_before_after")

# 3 ── Age distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(real_age["Age"], bins=26, kde=True, color=PAL[0], ax=ax)
ax.axvline(real_age["Age"].median(), color=PAL[1], ls="--", label=f"Median = {real_age['Age'].median():.0f}")
ax.set_title("Customer Age Distribution", fontweight="bold"); ax.legend()
save(fig, "03_age_distribution")

# 4 ── Purchase amount distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(real_amt["Purchase_Amount"], bins=40, kde=True, color=PAL[2], ax=ax)
ax.axvline(real_amt["Purchase_Amount"].median(), color=PAL[1], ls="--",
           label=f"Median = {real_amt['Purchase_Amount'].median():,.0f}")
ax.xaxis.set_major_formatter(inr)
ax.set_title("Purchase Amount Distribution", fontweight="bold"); ax.legend()
save(fig, "04_purchase_amount_distribution")

# 5 ── Customers & revenue by city
city = real_amt[real_amt["City"] != "Unknown"].groupby("City")["Purchase_Amount"].agg(["count", "sum", "mean"]).sort_values("sum", ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(x=city.index, y=city["sum"] / 1e6, ax=axes[0], color=PAL[0])
axes[0].set_title("Total Purchase Amount by City", fontweight="bold"); axes[0].set_ylabel("Millions")
sns.barplot(x=city.index, y=city["mean"], ax=axes[1], color=PAL[3])
axes[1].set_title("Average Purchase Amount by City", fontweight="bold"); axes[1].set_ylabel("Average")
axes[1].yaxis.set_major_formatter(inr)
for a in axes: a.set_xlabel("")
save(fig, "05_city_revenue")

# 6 ── Gender split and feedback scores
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
g = df["Gender"].value_counts()
axes[0].pie(g, labels=g.index, autopct="%1.1f%%", pctdistance=0.78, colors=PAL[:3], startangle=90, wedgeprops=dict(width=0.45))
axes[0].set_title("Gender Split (Unknown = missing in source)", fontweight="bold")
fb = df["Feedback_Score"].dropna().astype(int).value_counts().sort_index()
sns.barplot(x=fb.index, y=fb.values, ax=axes[1], color=PAL[4])
axes[1].set_title("Feedback Score Distribution", fontweight="bold"); axes[1].set_xlabel("Score"); axes[1].set_ylabel("Customers")
save(fig, "06_gender_and_feedback")

# 7 ── Signups over time
s = df.set_index("Signup_Date").resample("QE").size().iloc[:-1]   # drop final partial quarter (data ends 8 Oct 2025)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(s.index, s.values, marker="o", color=PAL[0], lw=2)
ax.fill_between(s.index, s.values, alpha=0.15, color=PAL[0])
ax.set_title("New Customer Signups per Quarter (complete quarters only)", fontweight="bold"); ax.set_ylabel("Signups"); ax.set_ylim(0, s.max()*1.15)
save(fig, "07_signups_over_time")

# 8 ── Age group vs spending
bins = [17, 25, 35, 45, 55, 65, 100]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]
both = df[~df["Age_Imputed"] & ~df["Purchase_Amount_Imputed"]].copy()
both["Age_Group"] = pd.cut(both["Age"], bins=bins, labels=labels)
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=both, x="Age_Group", y="Purchase_Amount", color="#CFE0EE", ax=ax)
ax.yaxis.set_major_formatter(inr)
ax.set_title("Purchase Amount by Age Group", fontweight="bold"); ax.set_xlabel("")
save(fig, "08_age_group_spending")

# 9 ── Feedback vs spending, correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fa = both.dropna(subset=["Feedback_Score"])
sns.boxplot(data=fa, x="Feedback_Score", y="Purchase_Amount", color="#CFE0EE", ax=axes[0])
axes[0].yaxis.set_major_formatter(inr)
axes[0].set_title("Purchase Amount by Feedback Score", fontweight="bold")
corr = fa[["Age", "Purchase_Amount", "Feedback_Score"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=axes[1], cbar=False)
axes[1].set_title("Correlation Matrix", fontweight="bold")
save(fig, "09_feedback_and_correlation")

# 10 ── Combined dashboard
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)
fig.suptitle("Customer Sales Data - Dashboard (cleaned data)", fontsize=20, fontweight="bold", y=0.99)

ax = fig.add_subplot(gs[0, 0])
sns.histplot(real_age["Age"], bins=26, kde=True, color=PAL[0], ax=ax); ax.set_title("Age Distribution", fontweight="bold")
ax = fig.add_subplot(gs[0, 1])
sns.histplot(real_amt["Purchase_Amount"], bins=40, kde=True, color=PAL[2], ax=ax)
ax.xaxis.set_major_formatter(inr); ax.set_title("Purchase Amount Distribution", fontweight="bold")
ax = fig.add_subplot(gs[0, 2])
sns.barplot(x=city.index, y=city["sum"] / 1e6, ax=ax, color=PAL[0])
ax.set_title("Total Purchases by City (millions)", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("Millions"); ax.tick_params(axis="x", rotation=30)
ax = fig.add_subplot(gs[1, 0])
ax.pie(g, labels=g.index, autopct="%1.1f%%", pctdistance=0.78, colors=PAL[:3], startangle=90, wedgeprops=dict(width=0.45))
ax.set_title("Gender Split", fontweight="bold")
ax = fig.add_subplot(gs[1, 1])
sns.barplot(x=fb.index, y=fb.values, ax=ax, color=PAL[4]); ax.set_title("Feedback Scores", fontweight="bold")
ax = fig.add_subplot(gs[1, 2])
ax.plot(s.index, s.values, marker="o", color=PAL[0]); ax.set_title("Signups per Quarter", fontweight="bold"); ax.set_ylim(0, s.max()*1.15)
ax.tick_params(axis="x", rotation=30)
fig.savefig(f"{OUT}/00_dashboard.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# key numbers for the summary
print("customers:", len(df))
print("avg purchase (real):", round(real_amt["Purchase_Amount"].mean()), "median:", real_amt["Purchase_Amount"].median())
print("top city by revenue:\n", city.round(0))
print("corr:\n", corr.round(3))
print("signup peak quarter:", s.idxmax().date(), s.max())
print("avg feedback:", round(df["Feedback_Score"].mean(), 2))
