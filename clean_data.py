import pandas as pd, numpy as np, json

import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "data", "raw", "messy_customer_sales_data.csv")
OUT = os.path.join(BASE, "data", "cleaned", "cleaned_customer_sales_data.csv")
log = {}

# Read everything as text first so nothing is silently coerced (e.g. phone leading zeros)
df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
log["raw_rows"] = len(df)

# ---------------------------------------------------------------- STEP 1: whitespace + blanks
df = df.apply(lambda c: c.str.strip())
df = df.replace("", np.nan)

# ---------------------------------------------------------------- STEP 2: inconsistent formatting
# Gender: m / M / male / MALE ...  ->  Male / Female
df["Gender"] = df["Gender"].str.lower().map(
    {"m": "Male", "male": "Male", "f": "Female", "female": "Female"})

# City: mixed case + padding -> Title Case
df["City"] = df["City"].str.title()

# Country: India / india / InDia / IND -> India
df["Country"] = df["Country"].str.lower().map({"india": "India", "ind": "India"})

# Name: strip titles (Mr., Mrs., Ms., Dr.) and suffixes (MD, DVM, DDS, PhD, Jr., Sr.)
df["Name"] = (df["Name"]
              .str.replace(r"^(Mr|Mrs|Ms|Miss|Dr)\.?\s+", "", regex=True)
              .str.replace(r"\s+(MD|DVM|DDS|PhD|Jr\.?|Sr\.?|II|III|IV)$", "", regex=True)
              .str.strip())

# Phone: stored as text; numbers that lost leading zeros are left-padded to 10 digits
log["phones_padded"] = int((df["Phone_Number"].str.len() < 10).sum())
df["Phone_Number"] = df["Phone_Number"].str.zfill(10)

# Email: lowercase
df["Email"] = df["Email"].str.lower()

# ---------------------------------------------------------------- STEP 3: data types
df["Age"] = pd.to_numeric(df["Age"].str.replace(r"\s*years", "", regex=True), errors="coerce")
df["Purchase_Amount"] = pd.to_numeric(df["Purchase_Amount"], errors="coerce")
df["Feedback_Score"] = pd.to_numeric(df["Feedback_Score"], errors="coerce")
df["Signup_Date"] = pd.to_datetime(df["Signup_Date"], errors="coerce")
df["Last_Purchase_Date"] = pd.to_datetime(df["Last_Purchase_Date"], errors="coerce")

# ---------------------------------------------------------------- STEP 4: duplicates
# After standardising, a customer is identified by phone + email (200 repeated rows).
# Rather than just dropping, consolidate: keep first non-null value per column in each group.
before = len(df)
df = df.groupby(["Phone_Number", "Email"], as_index=False, sort=False).first()
log["duplicates_removed"] = before - len(df)
df = df[["Customer_ID", "Name", "Gender", "Age", "City", "Signup_Date", "Last_Purchase_Date",
         "Purchase_Amount", "Feedback_Score", "Email", "Phone_Number", "Country"]]

# ---------------------------------------------------------------- STEP 5: outliers / invalid values
# Age: valid adult range 18-90 (found 3, -10, 250 ...)
bad_age = df["Age"].notna() & ~df["Age"].between(18, 90)
log["age_outliers"] = int(bad_age.sum())
df.loc[bad_age, "Age"] = np.nan

# Purchase_Amount: <=0 is invalid; upper bound via IQR fence on the valid positives
pos = df["Purchase_Amount"].where(df["Purchase_Amount"] > 0)
q1, q3 = pos.quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
bad_amt = df["Purchase_Amount"].notna() & ((df["Purchase_Amount"] <= 0) | (df["Purchase_Amount"] > upper))
log["amount_outliers"] = int(bad_amt.sum())
log["amount_upper_fence"] = float(upper)
df.loc[bad_amt, "Purchase_Amount"] = np.nan

# ---------------------------------------------------------------- STEP 6: missing values
# Flags first, so imputed values can always be told apart from real ones
df["Age_Imputed"] = df["Age"].isna()
df["Purchase_Amount_Imputed"] = df["Purchase_Amount"].isna()
df["Customer_ID_Missing"] = df["Customer_ID"].isna()
log["missing_before_fill"] = df.isna().sum().to_dict()

df["Age"] = df["Age"].fillna(df["Age"].median()).round().astype(int)
df["Purchase_Amount"] = df["Purchase_Amount"].fillna(df["Purchase_Amount"].median())
df["Gender"] = df["Gender"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["Country"] = df["Country"].fillna("India")        # every known city is Indian; 100% of known values = India
# Customer_ID cannot be recovered -> surrogate key so the column is a usable unique key
miss = df["Customer_ID"].isna()
df.loc[miss, "Customer_ID"] = ["UNK" + str(i + 1).zfill(4) for i in range(miss.sum())]
df["Feedback_Score"] = df["Feedback_Score"].astype("Int64")   # left blank on purpose (see report)

# ---------------------------------------------------------------- STEP 7: logical date check
df["Date_Inconsistent"] = df["Last_Purchase_Date"].notna() & (df["Last_Purchase_Date"] < df["Signup_Date"])
log["last_purchase_before_signup"] = int(df["Date_Inconsistent"].sum())

df = df.sort_values("Customer_ID").reset_index(drop=True)
df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
log["clean_rows"] = len(df)
print(json.dumps(log, indent=2, default=str))
