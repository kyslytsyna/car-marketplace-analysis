import pandas as pd
import numpy as np
import unicodedata
import re

df = pd.read_csv('data/automarket_autos.csv')
pd.set_option('display.max_columns', None)

# Initial inspection
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.describe(include='all'))

# Save original row count for later comparison
orig_rows = df.shape[0] 

# Rename columns to English
df.rename(columns={
    "Cena": "Price",
    "Rok výroby": "Year",
    "Stav": "Condition",
    "Najazdené": "Mileage",
    "Palivo": "Fuel",
    "Objem": "Engine",
    "Výkon": "Power",
    "Prevodovka": "Transmission",
    "Karoséria": "Body",
    "Pohon": "Drive",
    "Farba": "Color",
    "Norma": "EmissionStandard"
}, inplace=True)

# Check for missing values
print("Missing values per column:")
print(df.isna().sum()) 

# Check for rows with missing Title
print(df[df['Title'].isna()])  

# Drop rows from non-detail listing pages (e.g., 'https://kia-cee-d-sw.autobazar.sk/')
# that have no specific ad ID/path, which also results in missing 'Title'
df = df[df['Title'].notna()].copy()

# Drop missing target values
df = df[df['Price'].notna()].copy()
print(df.isna().sum())

# Check for duplicates
print("Number of duplicate rows:", df.duplicated().sum())

# Normalize text: remove accents, convert to lowercase
def normalize_text(text):
    if pd.isna(text):
        return text
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower() 

text_cols = ["Title","Condition", "Fuel", "Transmission", "Body", "Drive", "Color", "EmissionStandard"]
for col in text_cols:
    df[col] = df[col].apply(normalize_text)

# Prepare for extracting Brand from Title
# Whitelist of real car brands:
raw_brands = [
    "Skoda", "Kia", "Mercedes", "Volkswagen", "Tesla", "Ford", "Audi", "BMW", "Volvo",
    "Hyundai", "Subaru", "Peugeot", "Land Rover", "Mitsubishi", "Seat", "Mini", "Rolls-Royce",
    "Opel", "Suzuki", "Citroën", "Porsche", "Chevrolet", "Alfa Romeo", "Toyota", "Renault",
    "Fiat", "Nissan", "Iveco", "Cadillac", "Dongfeng", "Jeep", "Mazda", "Polestar", "Cupra",
    "Lexus", "Jaguar", "Maserati", "Bentley", "Infiniti", "Dodge", "Honda", "Dacia", "MG",
    "SsangYong", "Lada", "Smart", "DS", "GMC", "Chrysler", "Isuzu", "MAN", "Abarth", "Mahindra",
    "Ferrari", "Voyah", "Piaggio", "Tatra", "Daewoo", "Aston Martin", "Hongqi", "Oldsmobile",
    "Buick", "Lincoln", "Simca", "Alpina", "Lamborghini"
]

brands_norm = {normalize_text(b): b for b in raw_brands}
brands_norm["vw"] = "Volkswagen"
brands_norm["skodu"] = "Skoda"
brands_norm["rolls royce"] = "Rolls-Royce"
# Tokenize: split on whitespace
def tokenize(text):
    return normalize_text(text).split()

# Brand‐finder: check each token or adjacent token pair
def find_brand(title):
    tokens = tokenize(title)
    # Check single‐word brands first
    for tok in tokens:
        if tok in brands_norm:
            return brands_norm[tok]
    # Then check two‐word combos for multiword makes
    for i in range(len(tokens)-1):
        combo = f"{tokens[i]} {tokens[i+1]}"
        if combo in brands_norm:
            return brands_norm[combo]
    return "Unknown"

# Apply to DataFrame
df["Brand"] = df["Title"].apply(find_brand)

print(df["Brand"].value_counts())
print(df[df["Brand"] == "Unknown"])
print("Unknown brands:", (df["Brand"] == "Unknown").sum())

# Drop them in place
df = df[df["Brand"] != "Unknown"].reset_index(drop=True)

# Convert numeric columns to appropriate types
df["Price"] = df["Price"].str.replace('€', '', regex=False).str.replace(' ', '').str.replace(',', '.').astype(float)
df["Year"] = df["Year"].str.extract(r'(?:\d{1,2}/)?(\d{4})')
df["Year"] = pd.to_numeric(df['Year'], errors='coerce')
df["Mileage_km"] = (
    df["Mileage"]
      .str.replace(" km", "", regex=False)
      .str.replace(r"\s+", "", regex=True)
)
df["Mileage_km"] = pd.to_numeric(df["Mileage_km"], errors="coerce")
df = df.drop(columns=["Mileage"]) # Drop the raw string column
df[["Engine_cc","Engine_l"]] = df["Engine"].str.extract(
    r"(\d+)\s*cm³\s*\(\s*([\d\.]+)\s*l\)"
).astype(float)
df = df.drop(columns=["Engine"]) # Drop the raw string column
df[["Power_kW", "Power_PS"]] = df["Power"].str.extract(
    r"(?i)(\d+(?:[\.,]\d+)?)\s*kW\s*\(\s*(\d+)\s*PS\s*\)"
)
df["Power_kW"] = df["Power_kW"].str.replace(",", ".").astype(float)
df["Power_PS"] = df["Power_PS"].str.replace(",", ".").astype(float)
df = df.drop(columns=["Power"]) # Drop the raw string column

# Quick check
print(df.describe(include='all'))

# ---Audit the target (Price)---
cols_view = ["Title","Price","Year","Mileage_km","Fuel","URL_canon"]

print("Top 15 most expensive:")
print(df.nlargest(15, "Price")[cols_view])

print("\nBottom 15 cheapest:")
print(df.nsmallest(15, "Price")[cols_view])

q = df["Price"].quantile([0.99, 0.995, 0.999])
print("\nPrice quantiles:\n", q)

# Observations
# 1.Top outliers
# -Ads like “aixam mega s8 testovaci inzerat” - clear test entries.
# -“Cupra Formentor … 439111 €” - typo (probably 43,911 €).
# -These should be dropped or corrected.
# 2.Normal high end
# -Bentley, Lamborghini, Ferrari, Porsche around 200k–300k € - realistic.
# -0.995 quantile ≈ 106k, and 0.999 ≈ 178k - that’s probably where the true “supercar tail” starts.
# 3.Bottom outliers
# -“plechove disky” (just rims), “ponukame … na prenajom” (rental ads), “rozpredam na diely” (selling parts) - these are not cars.
# -Prices like 20 € - meaningless for car price prediction,so they should be dropped.

keywords = [
    "testovaci", "testovací", 
    "rozpredam", "rozpredám", 
    "prenajom", "prenájom", 
    "disky", "kolesa",
    "versatool", "krov", 
    "pila", "nastavec", 
    "motorová píla", " kosačka",
    "fukar"
]
# Filter out rows with these keywords in Title or Price < 200 or > 315000
mask_keywords = df["Title"].str.contains("|".join(keywords), case=False, na=False)
mask_price = (df["Price"] < 200) | (df["Price"] > 315000)
df_filtered = df[~(mask_keywords | mask_price)].copy()

# Inspect dropped rows
dropped = df[mask_keywords | mask_price]
print(dropped[["Title", "Price", "Year"]].head(20))
print(len(dropped), "rows dropped due to keywords or price limits.")

# ---Audit of Year---
print("Year summary:")
print(df_filtered["Year"].describe())

print("\nUnique years (sorted):")
print(sorted(df_filtered["Year"].dropna().unique()))

print(df_filtered[df_filtered["Year"].isna()].head())
print("Number of NaNs in 'Year': ",df_filtered["Year"].isna().sum())

# Fill missing Year values for 'nove' (new) condition
# Use the most common Year for 'nove' condition
# Build the mask BEFORE filling
to_fill_mask = (df_filtered["Condition"] == "nove") & (df_filtered["Year"].isna())
n_missing_before = int(to_fill_mask.sum())
print("Missing 'Year' for 'nove' before:", n_missing_before)

# Fill only those
most_common_new_year = df_filtered.loc[df_filtered["Condition"]=="nove", "Year"].mode()[0]
df_filtered.loc[to_fill_mask, "Year"] = most_common_new_year

# Verify fills
n_missing_after = int(df_filtered.loc[df_filtered["Condition"]=="nove", "Year"].isna().sum())
print("Missing 'Year' for 'nove' after:", n_missing_after)
print("Actually filled:", n_missing_before - n_missing_after)

# Drop rows with still missing Year
before = df_filtered.shape[0]
df_filtered = df_filtered[df_filtered["Year"].notna()].copy()
after = df_filtered.shape[0]

print(f"Dropped {before - after} rows with missing Year")

# Data correction note:
# During inspection, I found some obvious mistakes in scraped values 
# (e.g., car Year listed as 2026 instead of 2022).
# Where I could confirm the correct value directly from the original ad, 
# I manually corrected it in the dataset. 
# If the correct value could not be confirmed with certainty, the row was dropped.

# ---Audit of Mileage_km---
print("Mileage_km summary:")
print(df_filtered["Mileage_km"].describe())
print("Number of missing values in 'Mileage_km': ",df_filtered["Mileage_km"].isna().sum())
print(df_filtered[df_filtered["Mileage_km"].isna()].head())

# Fill missing mileage with 0 for new cars
df_filtered.loc[(df_filtered["Condition"]=="nove") & (df_filtered["Mileage_km"].isna()), "Mileage_km"] = 0

# Check how many NaNs are still left
print("Remaining NaN in Mileage_km:", df_filtered["Mileage_km"].isna().sum())

# Drop rows with still missing Mileage_km
before = df_filtered.shape[0]
df_filtered = df_filtered[df_filtered["Mileage_km"].notna()].copy()
after = df_filtered.shape[0]

print(f"Dropped {before - after} rows with missing Mileage_km")

# Check for odd values in Mileage_km
# Check the top 20 largest mileage values
print("\nTop 20 highest mileage:")
print(df_filtered.sort_values("Mileage_km", ascending=False)[["Title", "Mileage_km", "Year"]].head(20))

# Define threshold for truly impossible mileage
MAX_REALISTIC = 3_000_000  # Anything over 3 million km is almost certainly invalid

# Flag ads exceeding that limit
mask_implausible = df_filtered["Mileage_km"] > MAX_REALISTIC
print("Implausible mileage rows (>3M km):", int(mask_implausible.sum()))
print(df_filtered.loc[mask_implausible, ["Title", "Mileage_km", "URL_canon"]].head())

# Drop those implausible entries
df_filtered = df_filtered[~mask_implausible].copy()

# Keep a record of extremely high but valid mileages (e.g., >700k km) since they belong to vans ("dodávky"),
# where such extreme mileage is still realistic.
mask_high = df_filtered["Mileage_km"] > 700_000
print("High but valid mileage rows (>800k km):", int(mask_high.sum()))

# Check the bottom 20 (to catch negatives or weird very small non-zero values)
print("\nLowest 20 mileage values:")
print(df_filtered.sort_values("Mileage_km", ascending=True)[["Title", "Mileage_km", "Year"]].head(20))

mask_suspicious_low_used = (
    (df_filtered["Condition"] == "pouzivane") &
    (df_filtered["Year"] < 2023) &
    (df_filtered["Mileage_km"] < 200)
)

print(df_filtered.loc[mask_suspicious_low_used, ["Title", "Year", "Mileage_km", "Price", "URL_canon"]].head(20))
print("Suspicious low mileage used cars:", mask_suspicious_low_used.sum())

# Drop suspicious low mileage used cars
before = df_filtered.shape[0]
df_filtered = df_filtered.loc[~mask_suspicious_low_used].copy()
after = df_filtered.shape[0]
print(f"Dropped {before - after} rows; new shape: {df_filtered.shape}")

# ---Engine audit---
print("Engine_cc.describe():")
print(df_filtered["Engine_cc"].describe())

print("\nEngine_l.describe():")
print(df_filtered["Engine_l"].describe())

print("\nBiggest Engine_l (top 10):")
print(df_filtered.sort_values("Engine_l", ascending=False)[["Title","Engine_l","Engine_cc","URL_canon"]].head(10))

print("\nSmallest Engine_l (top 10):")
print(df_filtered.sort_values("Engine_l", ascending=True)[["Title","Engine_l","Engine_cc","URL_canon"]].head(10))

print("\nBiggest Engine_cc (top 10):")
print(df_filtered.sort_values("Engine_cc", ascending=False)[["Title","Engine_cc","Engine_l","URL_canon"]].head(10))

print("\nSmallest Engine_cc (top 10):")
print(df_filtered.sort_values("Engine_cc", ascending=True)[["Title","Engine_cc","Engine_l","URL_canon"]].head(10))

#Observations: 
#1. more then 10 l is definitely a mistake, so can be dropped 
#2. 0 l have electromotors, so other than electromotor with 0 l is a mistake

# Filter out engines larger than 10 liters
mask_too_big = df_filtered["Engine_l"] > 10
print("Rows with Engine_l > 10:", int(mask_too_big.sum()))
# Fill those with NaN
df_filtered.loc[mask_too_big, ["Engine_l","Engine_cc"]] = np.nan

mask_zero = df_filtered["Engine_l"] == 0
mask_zero_electric = mask_zero & (df_filtered["Fuel"].str.contains("elektro", case=False, na=False))
mask_zero_non_electric = mask_zero & ~mask_zero_electric

print("Zero-litre electric:", int(mask_zero_electric.sum()))
print("Zero-litre non-electric (to fix):", int(mask_zero_non_electric.sum()))
print(df_filtered.loc[mask_zero_non_electric, ["Title", "Engine_l","Engine_cc", "Fuel", "URL_canon"]].head())
# fix only the mistakes
df_filtered.loc[mask_zero_non_electric, ["Engine_l","Engine_cc"]] = np.nan

# most values < 0.6 are parsing errors too (microcars like Smart, Kei cars have 0.6-1.0 l)
mask_too_small = (df_filtered["Engine_l"] > 0) & (df_filtered["Engine_l"] < 0.6)
print(len(df_filtered[mask_too_small]), "rows with too small Engine_l") 
df_filtered.loc[mask_too_small, ["Engine_l","Engine_cc"]] = np.nan

# Plan: Later fill NaN with median engine size per brand/model, 
# or leave as NaN for ML algorithms that can handle missing values

# --- Power audit ---
print("Power_kW.describe():")
print(df_filtered["Power_kW"].describe())

print("\nPower_PS.describe():")
print(df_filtered["Power_PS"].describe())

print("\nTop 10 most powerful cars:")
print(df_filtered.sort_values("Power_kW", ascending=False)[["Title","Power_kW","Power_PS","URL_canon"]].head(10))

print("\nLowest 10 least powerful cars:")
print(df_filtered.sort_values("Power_kW", ascending=True)[["Title","Power_kW","Power_PS","URL_canon"]].head(10))

# Flag suspicious high power for small engines
is_electrified = (
    df_filtered["Fuel"].str.contains("elektr", case=False, na=False) |
    df_filtered["Fuel"].str.contains("hybrid", case=False, na=False)
)
# suspicious if:
# - small engine (< 2.0 L) 
# - very high power (> 300 kW)
# - NOT electric / hybrid
mask_suspicious_power = (
    (df_filtered["Engine_l"] < 2) &
    (df_filtered["Power_kW"] > 300) &
    ~df_filtered["Fuel"].str.contains("elektro|hybrid", case=False, na=False)
)

print("Suspicious small-engine high-power cars:", mask_suspicious_power.sum())
print(df_filtered.loc[mask_suspicious_power, 
      ["Brand", "Title","Engine_l","Power_kW","Fuel","URL_canon"]])


df_filtered.loc[mask_suspicious_power, ["Power_kW","Power_PS"]] = np.nan

# fake if:
# - power > 300 kW
# - brand is one of the common budget brands
mask_fake_high_power = (
    (df_filtered["Power_kW"] > 300) &
    (df_filtered["Brand"].str.lower().isin(["fiat", "skoda", "kia", "peugeot", "renault"]))
)

print("Clearly fake:", mask_fake_high_power.sum())
print(df_filtered.loc[mask_fake_high_power, ["Brand","Title","Power_kW","Engine_l","Fuel","URL_canon"]])

# Fix
df_filtered.loc[mask_fake_high_power, ["Power_kW","Power_PS"]] = np.nan


# --- Fuel cleaning ---
# Show rows with missing Fuel
missing_fuel = df_filtered[df_filtered["Fuel"].isna()]
print("Cars with missing Fuel:", missing_fuel.shape[0])
print(missing_fuel[["Brand","Title","Price","Engine_l","Power_kW","Year","URL_canon"]])

# Drop rows with missing Fuel
before = df_filtered.shape[0]
df_filtered = df_filtered[df_filtered["Fuel"].notna()].copy()
after = df_filtered.shape[0]
print(f"Dropped {before - after} rows with missing Fuel; new shape: {df_filtered.shape}")

# Check for suspicious Fuel values
print("Unique Fuel values:")
print(df_filtered["Fuel"].unique())

fuel_map = {
    "benzin": "Petrol",
    "diesel": "Diesel",
    "elektrina  diesel": "Hybrid Diesel",
    "elektromotor": "Electric",
    "hybrid": "Hybrid Petrol",
    "plugin hybrid": "Plug-in Hybrid",
    "lpg  benzin": "LPG",
    "cng  benzin": "CNG",
    "hybrid benzin": "Hybrid Petrol",
    "1": "Unknown"
}

df_filtered["Fuel_clean"] = df_filtered["Fuel"].map(fuel_map).fillna("Other")
print(df_filtered["Fuel_clean"].value_counts())

def collapse_fuel(x):
    if x in ["Petrol", "Diesel", "Electric", "Plug-in Hybrid"]:
        return x
    elif "Hybrid" in x:  # Hybrid Petrol, Hybrid Diesel
        return "Hybrid"
    elif x in ["LPG", "CNG", "Other", "Unknown"]:
        return "Alternative Fuels"
    else:
        return "Other"

df_filtered["Fuel_collapsed"] = df_filtered["Fuel_clean"].apply(collapse_fuel)

print(df_filtered["Fuel_collapsed"].value_counts())

# Drop the original Fuel column and rename the cleaned one
df_filtered["Fuel"] = df_filtered["Fuel_clean"]
df_filtered = df_filtered.drop(columns=["Fuel_clean"])

# There were only 14 rows with missing Fuel values.
# I manually inspected each ad on the website:
# - For cases where I was 100% certain about the correct fuel type, I filled it in.
# - For ambiguous cases where the fuel type could not be reliably determined, I dropped the rows.

# --- Condition cleaning ---
missing_condition = df_filtered[df_filtered["Condition"].isna()]
print("Cars with missing Condition:", missing_condition.shape[0])
print(missing_condition[["Brand","Title","Price","Mileage_km","Engine_l","Power_kW","Year","URL_canon"]])

print(df_filtered.loc[df_filtered["Condition"].isna(), "Title"].head(20).to_list())


# Condition imputation from Title keywords
is_new  = df_filtered["Title"].str.contains(r"\b(nove|novy|nove vozidlo|new)\b", case=False, regex=True)
is_demo = df_filtered["Title"].str.contains(r"\b(demo|predvadzacie|predvadzacie vozidlo)\b", case=False, regex=True)

# Only fill where Condition is missing
mask_na = df_filtered["Condition"].isna()

# Fill explicit cases first
df_filtered.loc[mask_na & is_new,  "Condition"] = "nove"
df_filtered.loc[mask_na & is_demo, "Condition"] = "demo"

mask_low_km_price = (
    mask_na
    & df_filtered["Mileage_km"].between(0, 100, inclusive="both")
    & (df_filtered["Price"] >= 25_000)
)

df_filtered.loc[mask_low_km_price, "Condition"] = "demo"

# Everything else still missing - assume used
still_na = df_filtered["Condition"].isna()
df_filtered.loc[still_na, "Condition"] = "pouzivane"

# Quick audit
print("Filled 'nove' from title:", int((mask_na & is_new).sum()))
print("Filled 'demo' from title:", int((mask_na & is_demo).sum()))
print("Filled 'demo' from low mileage:", int(mask_low_km_price.sum()))
print("Remaining missing after fill:", int(df_filtered["Condition"].isna().sum()))
print(df_filtered["Condition"].value_counts(dropna=False))


# --- Engine_l and Engine_cc cleaning ---
print(df_filtered.isna().sum())
# Engine imputation for EVs
mask_engine_na = df_filtered["Engine_cc"].isna()

# detect electric or plug-in hybrid
mask_electric = df_filtered["Fuel"].str.contains(
    r"(?:elektro|electric|ev|plug\-in)", case=False, na=False
)

# fill missing Engine for EVs
df_filtered.loc[mask_engine_na & mask_electric, "Engine_cc"] = 0
df_filtered.loc[mask_engine_na & mask_electric, "Engine_l"]  = 0.0

print("Filled EV engines:", (mask_engine_na & mask_electric).sum())
print("Remaining missing Engine after EV fix:", df_filtered["Engine_cc"].isna().sum())

# fill missing Engine_l and Engine_cc for non-EVs
# Use median per Brand, Year and Power_kW
engine_map = (
    df_filtered.dropna(subset=["Engine_cc"])
    .groupby(["Brand", "Year", "Power_kW"])
    .agg({"Engine_cc": "first", "Engine_l": "first"})
    .to_dict("index")
)


def impute_engine(row):
    if pd.isna(row["Engine_cc"]):
        key = (row["Brand"], row["Year"], row["Power_kW"])
        if key in engine_map:
            row["Engine_cc"] = engine_map[key]["Engine_cc"]
            row["Engine_l"]  = engine_map[key]["Engine_l"]
    return row

df_filtered = df_filtered.apply(impute_engine, axis=1)

print("Remaining Engine_cc missing:", df_filtered["Engine_cc"].isna().sum())
print("Remaining Engine_l missing:", df_filtered["Engine_l"].isna().sum())


# --- Engine binning ---
# Build a single numeric engine displacement (liters)
engine_l_eff = df_filtered["Engine_l"].copy()

cc_to_l = df_filtered["Engine_cc"].div(1000)
engine_l_eff = engine_l_eff.fillna(cc_to_l)

# EV mask (use both fuel text and literal 0 L)
is_ev = (
    df_filtered["Fuel"].str.contains(r"elektro|electric|ev|plug-?in", case=False, na=False)
    | (engine_l_eff == 0)
)

# Create the bin column with a default
df_filtered["Engine_bin"] = "Unknown"

# EV bin
df_filtered.loc[is_ev, "Engine_bin"] = "EV (0 L)"

# Numeric bins for non-EV where we have a valid liters value (>0)
mask_numeric = (~is_ev) & engine_l_eff.notna() & (engine_l_eff > 0)

bins = [0, 1.2, 1.8, 2.2, 3.0, np.inf]
labels = ["≤1.2 L", "1.2–1.8 L", "1.8–2.2 L", "2.2–3.0 L", ">3.0 L"]

df_filtered.loc[mask_numeric, "Engine_bin"] = pd.cut(
    engine_l_eff[mask_numeric],
    bins=bins,
    labels=labels,
    right=True,           # bin is right-closed: (a, b]
    include_lowest=False  # 0 is not included here (we already captured EV=0)
).astype(str)

# Make it an ordered categorical (nice for modeling/plots)
order = ["EV (0 L)", "≤1.2 L", "1.2–1.8 L", "1.8–2.2 L", "2.2–3.0 L", ">3.0 L", "Unknown"]
df_filtered["Engine_bin"] = pd.Categorical(df_filtered["Engine_bin"], categories=order, ordered=True)

# Quick audit
print(df_filtered["Engine_bin"].value_counts(dropna=False))


# --- Power_kW and Power_PS cleaning ---
# Both Power_kW and Power_PS are always missing together (never one without the other).
# Since these values are critical features for car pricing and performance,
# we drop rows where they are missing. This removes only ~195 rows (<1% of data),
# which is acceptable for keeping dataset consistency.
before = df_filtered.shape[0]
df_filtered = df_filtered.dropna(subset=["Power_kW", "Power_PS"])
after = df_filtered.shape[0]

print(f"Dropped rows with missing Power: {before - after}")
print(f"Remaining rows: {after}")

# Check for % of dropped rows 
print("Original rows:", orig_rows)
print("After cleaning:", df_filtered.shape[0])
print("Dropped rows:", orig_rows - df_filtered.shape[0])
print("Dropped %:", round((orig_rows - df_filtered.shape[0]) / orig_rows * 100, 2), "%")

# --- Transmission cleaning ---
print("Unique Transmission values (raw):")
print(df_filtered["Transmission"].dropna().unique())

print("\nValue counts:")
print(df_filtered["Transmission"].value_counts(dropna=False))

# Define a function to simplify Transmission values
def simplify_transmission(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower()
    if "manual" in x:
        return "manualna"
    if "automat" in x:
        return "automaticka"
    return np.nan   # catch weird stuff like 3073, 3074
        

df_filtered["Transmission_simple"] = df_filtered["Transmission"].apply(simplify_transmission)

print(df_filtered["Transmission_simple"].value_counts(dropna=False))

# Since transmission type (manual vs. automatic) is an important feature for car price prediction,  
# I decided to drop rows with missing values instead of filling them with 'unknown'.  
# This ensures model training is not biased by artificial categories and keeps data quality high. 
 
# Drop rows with missing Transmission
before = df_filtered.shape[0]
df_filtered = df_filtered.dropna(subset=["Transmission_simple"])
after = df_filtered.shape[0]

print(f"Dropped rows with missing Transmission: {before - after}")
print(f"Remaining rows: {after}")

# --- Body cleaning ---
print("Unique Body values (raw):")
print(df_filtered["Body"].dropna().unique())

# Define a mapping for Body types
body_mapping = {
    # Sedan group
    "sedan": "sedan",
    "limuzina": "sedan",
    "liftback": "sedan",
    "sedan  liftback": "sedan",

    # Hatchback group
    "hatchback": "hatchback",
    "suv  hatchback": "hatchback",

    # Combi / Wagon group
    "combi": "combi",
    "suv  combi": "combi",

    # SUV group
    "suv": "suv",
    "suv  offroad": "suv",

    # Coupe group
    "coupe": "coupe",
    "suv  coupe": "coupe",

    # Cabrio / Convertible group
    "cabrio": "cabrio",
    "roadster": "cabrio",
    "targa": "cabrio",
    "suv  cabrio": "cabrio",

    # Van / MPV group
    "van": "van/mpv",
    "mpv": "van/mpv",
    "minibus": "van/mpv",

    # Pickup group
    "pick up": "pickup",

    # Utility / Commercial group
    "dodavka": "utility",
    "skrina": "utility",
    "valnik": "utility",
    "chladiarenske": "utility",
    "mraziarenske": "utility",
    "kabina": "utility",
    "podvozok": "utility",

    # Other / Parsing errors
    "ine": "other",
    "10": "other"
}

# Apply mapping
df_filtered["Body_clean"] = df_filtered["Body"].map(body_mapping).fillna("unknown")

print("Original unique values:", df_filtered["Body"].nunique())
print("Cleaned unique values:", df_filtered["Body_clean"].nunique())
print(df_filtered["Body_clean"].value_counts(dropna=False))

# Drop the original Body column and rename the cleaned one
df_filtered["Body"] = df_filtered["Body_clean"]
df_filtered = df_filtered.drop(columns=["Body_clean"])


# --- Drive cleaning ---
print("Unique Drive values (raw):")
print(df_filtered["Drive"].dropna().unique())

# Define a mapping for Drive types
drive_map = {
    "predny": "FWD",
    "zadny": "RWD",
    "4x4": "AWD",
    "ine": "other"
}
df_filtered["Drive_clean"] = df_filtered["Drive"].map(drive_map).fillna("unknown")
print(df_filtered["Drive_clean"].value_counts(dropna=False))

# Replace original column with cleaned one
df_filtered["Drive"] = df_filtered["Drive_clean"]
df_filtered = df_filtered.drop(columns=["Drive_clean"])

# --- Color cleaning ---
print("Unique colors:", df_filtered["Color"].unique())
print("\nValue counts:")
print(df_filtered["Color"].value_counts(dropna=False))

# Define a mapping function for colors
def map_color(value):
    if pd.isna(value):
        return "unknown"
    v = value.lower()
    if "biela" in v:
        return "white"
    elif "cierna" in v or "black" in v:
        return "black"
    elif "siva" in v or "seda" in v:
        return "grey"
    elif "strieb" in v:
        return "silver"
    elif "modra" in v:
        return "blue"
    elif "cerv" in v or "bord" in v:
        return "red"
    elif "zelena" in v:
        return "green"
    elif "zlta" in v:
        return "yellow"
    elif "oran" in v:
        return "orange"
    elif "hned" in v:
        return "brown"
    elif "bezov" in v:
        return "beige"
    elif "fial" in v:
        return "purple"
    elif "ruzov" in v:
        return "pink"
    elif "zlat" in v:
        return "gold"
    else:
        return "unknown"

df_filtered["Color_clean"] = df_filtered["Color"].apply(map_color)
print(df_filtered["Color_clean"].value_counts())

# Define rare colors to merge
rare_colors = ["gold", "beige", "purple", "pink", "orange", "yellow"]

df_filtered["Color_clean"] = df_filtered["Color_clean"].replace(rare_colors, "other")

# Re-check counts
print(df_filtered["Color_clean"].value_counts())

# Replace original column with cleaned one
df_filtered["Color"] = df_filtered["Color_clean"]
df_filtered = df_filtered.drop(columns=["Color_clean"])

# --- Emission Standard cleaning ---
print("Unique Emission Standard values (raw):")
print(df_filtered["EmissionStandard"].dropna().unique())

# Get rif of junk values
em = df_filtered["EmissionStandard"]
em = em.replace({"207": np.nan, "": np.nan, "nan": np.nan})

# Collapse euro 6 sub-variants to 'euro 6'
em = (em
      .replace({
          "euro 6a": "euro 6",
          "euro 6b": "euro 6",
          "euro 6c": "euro 6",
          "euro 6d": "euro 6",
          "euro 6dtemp": "euro 6",
          "euro 6e": "euro 6",
          "eev": "euro 6"
      }))

# Keep only known labels; everything else → NaN for now
valid_labels = {"euro 2","euro 3","euro 4","euro 5","euro 6"}
em = em.where(em.isin(valid_labels), np.nan)

# Set EVs to 'ev' if missing Euro (optional but useful)
is_ev = df_filtered["Fuel"].str.contains("elektr", case=False, na=False)
em = np.where(is_ev & em.isna(), "ev", em)

# Final fill for true-missing
df_filtered["EmissionStandard"] = pd.Series(em, index=df_filtered.index).fillna("unknown")

# Quick audit
print(df_filtered["EmissionStandard"].value_counts(dropna=False))

# Normalize + ordinal encode

# Normalize to tidy labels (string categorical)
emap = {
    "euro 2": "Euro 2",
    "euro 3": "Euro 3",
    "euro 4": "Euro 4",
    "euro 5": "Euro 5",
    "euro 6": "Euro 6",
    "eev":    "Euro 6",   # treat EEV ≈ Euro 6
    "ev":     "EV",
    "unknown":"Unknown"
}

# Ensure lower-case base, then map → tidy labels
em_base = df_filtered["EmissionStandard"].astype(str).str.lower().str.strip()
df_filtered["EmissionStandard_clean"] = em_base.map(emap).fillna("Unknown")

print("EmissionStandard_clean counts:")
print(df_filtered["EmissionStandard_clean"].value_counts(dropna=False))

# Replace original with cleaned
df_filtered["EmissionStandard"] = df_filtered["EmissionStandard_clean"]

# Drop the helper column
df_filtered = df_filtered.drop(columns=["EmissionStandard_clean"])

print(df_filtered.isna().sum())  # Check for remaining NaNs

# Droop remaining missing values
before = df_filtered.shape[0]
df_filtered = df_filtered.dropna(subset=["Engine_cc", "Engine_l"])
after = df_filtered.shape[0]
dropped = before - after

print(f"Dropped rows with missing Engine_cc/Engine_l: {dropped}")
print(f"Remaining rows: {after}")
print(f"Dropped % of original: {round(dropped / df.shape[0] * 100, 2)} %")

# Create a Model column
# Define a mapping of top 30 brands to their common models
# pick top N brands
top_brands = df_filtered["Brand"].value_counts().head(30).index.tolist()
print("Top 30 brands:", top_brands)
brand_models = {
    "Skoda":        ["Octavia","Superb","Fabia","Kodiaq","Scala","Kamiq","Rapid","Roomster", "Citigo", "Enyaq", "Karoq", "Yeti"],
    "Volkswagen":   ["Golf","Passat","Tiguan","Polo","Arteon","Touran","Passat Variant","Caddy", "T-Roc", "ID.3", "ID.4", "ID. Buzz", "T-Cross", "Tiguan Allspace"],
    "Ford":         ["Focus","Fiesta","Kuga","Mondeo","EcoSport","Transit","Transit Custom", "Tourneo Connect", "Tourneo Custom", "Mustang", "Puma"],
    "Peugeot":      ["308", "3008", "2008", "208", "508", "5008", "206", "207", "Partner", "Expert", "Boxer", "Traveller"],
    "BMW":          ["Series 3","Series 5","X5","X3","X1","Series 1","Rad 5","Rad 3", "X6","X4","X7","Z4","i3","iX3","i4","iX"],
    "Mercedes":     ["A trieda", "C trieda", "E trieda", "S trieda", "GLC", "GLE","Benz C","Vito","Citan", "CLA", "GLA", "GLB", "GLS", "EQC", "EQB", "EQE", "EQS"],
    "Audi":         ["A3", "A4", "A6", "Q5", "Q7", "Q3", "Q8", "A1", "A5", "A8", "Q2", "Q4", "A2", "A7", "TT", "RS4", "RS5"],
    "Opel":         ["Astra", "Corsa", "Insignia", "Mokka", "Crossland", "Vivaro","Zafira", "Grandland", "Combo"],
    "Hyundai":      ["i30", "i20", "Tucson", "Kona", "Santa Fe", "i10", "i40", "iX35", "iX20", "Veloster", "Bayon", "Ioniq"],
    "Citroën":      ["C3", "C4", "C5", "Berlingo", "C1", "C2", "C6", "C8", "C-Zero", "DS3", "DS4", "DS5"],
    "Kia":          ["Ceed", "Sportage", "Rio", "Sorento", "Stonic", "Niro", "XCeed", "Picanto", "Optima", "Soul", "Stinger", "EV6", "Seltos"],
    "Renault":      ["Clio", "Megane", "Captur", "Talisman", "Scénic", "Kangoo","Trafic", "Master", "Kadjar", "Zoe", "Arkana"],
    "Fiat":         ["500", "Panda", "Tipo", "Punto", "500X", "Doblo","Ducato", "Fiorino", "Talento", "Fullback", "Punto Evo"],
    "Toyota":       ["Yaris", "Corolla", "RAV4", "C-HR", "Avensis", "Hilux","Proace", "Prius", "Land Cruiser", "Camry", "Aygo", "Verso", "CHR"],
    "Seat":         ["Leon", "Ibiza", "Arona", "Ateca", "Alhambra", "Toledo", "Altea", "Exeo", "Mii", "Tarraco", "Cupra"],
    "Suzuki":       ["Swift", "Vitara", "S-Cross", "Ignis", "SX4", "Grand Vitara", "Baleno", "Jimny", "Celerio", "Liana"],
    "Dacia":        ["Duster", "Sandero", "Logan", "Dokker", "Lodgy", "Spring", "Duster Pick-up", "Sandero Stepway"],
    "Mazda":        ["2", "3", "6", "CX-5", "CX-3", "CX-30", "MX-5", "CX-60", "CX-50", "CX-90", "CX-70"],
    "Volvo":        ["XC60", "XC90", "V40", "V60", "S60", "S90", "V90", "C30", "C70", "S40", "V50", "V70", "XC40"],
    "Nissan":       ["Qashqai", "Juke", "X-Trail", "Leaf", "Micra", "Navara", "Patrol", "Primera", "Almera", "Note", "NV200", "NV300", "NV400"],
    "Honda":        ["Civic", "Jazz", "CR-V", "HR-V", "Accord", "FR-V", "Insight", "CR-Z", "S2000"],
    "Land Rover":   ["Range Rover", "Discovery", "Freelander", "Defender", "Evoque", "Discovery Sport", "Velar", "Defender 90", "Defender 110"],
    "Mitsubishi":   ["Outlander", "ASX", "L200", "Pajero", "Space Star", "Galant", "Colt", "Eclipse Cross", "Lancer", "Mirage"],
    "Jeep":         ["Renegade", "Compass", "Wrangler", "Cherokee", "Grand Cherokee", "Patriot", "Liberty", "Commander", "Gladiator"],
    "Mini":         ["Cooper", "Clubman", "Countryman", "Convertible", "Paceman", "Roadster", "Mini Electric", "Mini John Cooper Works"],
    "Iveco":        ["Daily", "Eurocargo", "Stralis", "Trakker", "Eurotech", "Eurostar", "Massif", "New Daily"],
    "Subaru":       ["Forester", "Outback", "Impreza", "XV", "Legacy", "BRZ", "Levorg", "Tribeca", "Ascent"],
    "Lexus":        ["NX", "RX", "IS", "UX", "ES", "GS", "LS", "CT", "LC", "LX", "RX L", "RX F Sport"],
    "Jaguar":       ["XE", "XF", "F-Pace", "E-Pace", "I-Pace", "F-Type", "XJ", "S-Type", "X-Type", "F-Pace SVR"],
    "Alfa Romeo":   ["Giulia", "Stelvio", "Giulietta", "159", "Mito", "4C", "Spider", "Brera", "Alfa 6", "Alfa 75"],
    "Chevrolet":    ["Cruze", "Aveo", "Spark", "Camaro", "Captiva", "Orlando", "Trax", "Malibu", "Volt", "Bolt EV"]
}
brand_models_norm = {
    b: {m.lower(): m for m in models} for b, models in brand_models.items()
}

def extract_model_whitelist(title, brand):
    if brand not in brand_models_norm:
        return "Unknown"
    t = title.lower()
    for m_norm, raw in brand_models_norm[brand].items():
        # word-boundary-ish match; adjust if needed for hyphens/dots
        if re.search(rf"\b{re.escape(m_norm)}\b", t):
            return raw
    return "Unknown"

df_filtered["Model_simple"] = df_filtered.apply(
    lambda r: extract_model_whitelist(r["Title"], r["Brand"]), axis=1
)

# (Optional) collapse rare models within each brand to "Other"
keep_per_brand = 10
def collapse_rare(group):
    top = group.value_counts().head(keep_per_brand).index
    return group.where(group.isin(top), "Other")

df_filtered["Model_simple"] = (
    df_filtered.groupby("Brand")["Model_simple"].transform(collapse_rare)
)

# sanity check
print(df_filtered["Model_simple"].value_counts().head(50))
print("Unknown share:", (df_filtered["Model_simple"]=="Unknown").mean())

# Check for % of dropped rows 
print("Original rows:", orig_rows)
print("After cleaning:", df_filtered.shape[0])
print("Dropped rows:", orig_rows - df_filtered.shape[0])
print("Dropped %:", round((orig_rows - df_filtered.shape[0]) / orig_rows * 100, 2), "%")

# Save the cleaned DataFrame to a CSV file
df_filtered.to_csv("data/cleaned_automarket_autos.csv", index=False)


