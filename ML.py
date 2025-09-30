# Core
import pandas as pd
import numpy as np

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Statsmodels (statistical analysis)
from scipy.stats import loguniform, randint

# ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor


# Load preprocessed data
df = pd.read_csv("data/cleaned_automarket_autos.csv")

print(df.shape)
print(df.head())

# Drop obvious outliers (same logic as in EDA)
p_lo, p_hi = df["Price"].quantile([0.01, 0.99])
m_lo, m_hi = 100, df["Mileage_km"].quantile(0.995)

df_ml = df.loc[
    df["Price"].between(p_lo, p_hi) &
    df["Mileage_km"].between(m_lo, m_hi) &
    (df["Condition"].str.lower() != "havarovane")
].copy()

# Add Car_Age
df_ml["Car_Age"] = 2025 - df_ml["Year"]

# Add 5-year bins
df_ml["Year_bin"] = pd.cut(
    df_ml["Year"],
    bins=[1950, 1960, 1970, 1980, 1990, 2000, 2010, 2015, 2020, 2025, 2030],
    labels=[
        "1950–1959", "1960–1969", "1970–1979", "1980–1989",
        "1990–1999", "2000–2009", "2010–2014", "2015–2019",
        "2020–2024", "2025–2029"
    ],
    right=False  # left-closed, right-open
)
print(df_ml[["Year", "Year_bin", "Car_Age"]].head(10))

# Group brands by market segment
brand_to_segment = {
    # Luxury / Premium
    "Audi": "Luxury", "BMW": "Luxury", "Mercedes": "Luxury",
    "Porsche": "Luxury", "Jaguar": "Luxury", "Lexus": "Luxury",
    "Land Rover": "Luxury", "Maserati": "Luxury", "Bentley": "Luxury",
    "Aston Martin": "Luxury", "Ferrari": "Luxury", "Cadillac": "Luxury",
    "Lincoln": "Luxury", "Alpina": "Luxury", "Hongqi": "Luxury",
    
    # Upper Midrange
    "Volkswagen": "Upper Midrange", "Volvo": "Upper Midrange",
    "Mini": "Upper Midrange", "Cupra": "Upper Midrange",
    
    # Mainstream / Midrange
    "Skoda": "Midrange", "Kia": "Midrange", "Hyundai": "Midrange",
    "Toyota": "Midrange", "Ford": "Midrange", "Peugeot": "Midrange",
    "Renault": "Midrange", "Citroën": "Midrange", "Seat": "Midrange",
    "Mazda": "Midrange", "Honda": "Midrange", "Nissan": "Midrange",
    "Subaru": "Midrange",
    
    # Budget / Economy
    "Dacia": "Budget", "Fiat": "Budget", "Opel": "Budget",
    "Suzuki": "Budget", "Lada": "Budget", "SsangYong": "Budget",
    
    # Commercial / Utility
    "MAN": "Commercial", "Iveco": "Commercial", "Isuzu": "Commercial",
    "Piaggio": "Commercial",
    
    # US Brands
    "Jeep": "US SUV", "Dodge": "US SUV", "Chevrolet": "US SUV",
    "Chrysler": "US SUV", "Buick": "US SUV",
    
    # Electric Focused
    "Tesla": "Electric Focused", "Smart": "Electric Focused", "MG": "Electric Focused",
    
    # Other / Rare
    "DS": "Other", "Dongfeng": "Other", "Mahindra": "Other", "Simca": "Other",
    "Daewoo": "Other", "Infiniti": "Other",
    "Abarth": "Other", "Alfa Romeo": "Other",
}

# Apply mapping to DataFrame
df_ml["Brand_Segment"] = df_ml["Brand"].map(brand_to_segment).fillna("Other")

# Check results
print(df_ml["Brand_Segment"].value_counts())

# ============================================
# MODEL 1: Base Linear Regression
# ============================================

# Features
num_cols = ["Car_Age", "Mileage_km", "Power_kW"]
cat_cols = ["Fuel", "Body", "Brand_Segment"]
bin_cols = ["Transmission_simple"]
X_base = df_ml[num_cols + bin_cols + cat_cols]

# Log-transform the target
df_ml["Log_Price"] = np.log1p(df_ml["Price"])
y = df_ml["Log_Price"]

# Define preprocessing and modeling pipeline
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), bin_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])

pipe = Pipeline([
    ("prep", preprocess),
    ("model", LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42)
# Fit and evaluate model
pipe.fit(X_train, y_train)
y_pred_log = pipe.predict(X_test)

# Convert back to original scale
y_pred_price = np.expm1(y_pred_log)
y_test_price = np.expm1(y_test)

# Evaluation metrics
mae = mean_absolute_error(y_test_price, y_pred_price)
r2 = r2_score(y_test_price, y_pred_price)
print("Model 1")
print(f"MAE: {mae:.2f} €")
print(f"R²: {r2:.3f}")

# Also show metrics in log space for reference
mae_log = mean_absolute_error(y_test, y_pred_log)
r2_log  = r2_score(y_test, y_pred_log)
print(f"MAE (log space): {mae_log:.3f}")
print(f"R² (log space):  {r2_log:.3f}")

# ================================================
# MODEL 2: Linear Regression with Mileage_per_Year
# ================================================

# Create new feature
df_ml["Mileage_per_Year"] = df_ml["Mileage_km"] / df_ml["Car_Age"].replace(0, 1)

# New numeric features set
num_cols_v2 = ["Year","Mileage_per_Year", "Power_kW"]
X_base_v2 = df_ml[num_cols_v2 + bin_cols + cat_cols]

# Redefine preprocess and pipeline
preprocess_v2 = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols_v2),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), bin_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

pipe_v2 = Pipeline([
    ("prep", preprocess_v2),
    ("model", LinearRegression())
])

# Use the same target and split
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(X_base_v2, y, test_size=0.2, random_state=42)

pipe_v2.fit(X_train_v2, y_train_v2)
y_pred_log_v2 = pipe_v2.predict(X_test_v2)
y_pred_price_v2 = np.expm1(y_pred_log_v2)

mae_v2 = mean_absolute_error(np.expm1(y_test_v2), y_pred_price_v2)
r2_v2 = r2_score(np.expm1(y_test_v2), y_pred_price_v2)

print("Model 2")
print(f"MAE: {mae_v2:.2f} €")
print(f"R²: {r2_v2:.3f}")

# ============================================
# MODEL 3: RidgeCV with Age + Mileage_km + Mileage_per_Year
# ============================================

# Numeric features for Ridge (regularized to handle multicollinearity)
num_cols_v3 = ["Car_Age", "Mileage_km", "Mileage_per_Year", "Power_kW"]
X_base_v3 = df_ml[num_cols_v3 + bin_cols + cat_cols]

# Redefine preprocess and pipeline
preprocess_v3 = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_v3),
        ("bin", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), bin_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop"
)

# Cross-validated alpha on a log grid
alphas = np.logspace(-3, 3, 25)
pipe_v3 = Pipeline([
    ("prep", preprocess_v3),
    ("model", RidgeCV(alphas=alphas))
])

# Use the same split pattern as Model 2 for fair comparison
X_train_v3, X_test_v3, y_train_v3, y_test_v3 = train_test_split(
    X_base_v3, y, test_size=0.2, random_state=42
)

pipe_v3.fit(X_train_v3, y_train_v3)
y_pred_log_v3 = pipe_v3.predict(X_test_v3)
y_pred_price_v3 = np.expm1(y_pred_log_v3)

# Evaluate
mae_v3 = mean_absolute_error(np.expm1(y_test_v3), y_pred_price_v3)
r2_v3  = r2_score(np.expm1(y_test_v3), y_pred_price_v3)
mae_log_v3 = mean_absolute_error(y_test_v3, y_pred_log_v3)
r2_log_v3  = r2_score(y_test_v3, y_pred_log_v3)

print("Model 3")
print(f"MAE: {mae_v3:.2f} €")
print(f"R²:  {r2_v3:.3f}")
print(f"MAE (log): {mae_log_v3:.3f}")
print(f"R² (log):  {r2_log_v3:.3f}")

# Show chosen alpha
best_alpha = pipe_v3.named_steps["model"].alpha_
print(f"Ridge best alpha: {best_alpha:.4f}")

# Inspect top coefficients
def show_top_coefs(pipe, k=15):
    prep  = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]

    num_names = prep.transformers_[0][2]
    bin_ohe   = prep.named_transformers_["bin"]
    cat_ohe   = prep.named_transformers_["cat"]
    bin_names = bin_ohe.get_feature_names_out(bin_cols)
    cat_names = cat_ohe.get_feature_names_out(cat_cols)
    feat_names = np.r_[num_names, bin_names, cat_names]

    coefs = model.coef_.ravel()
    coef_df = (pd.DataFrame({"feature": feat_names, "coef": coefs})
                 .sort_values("coef", ascending=False))
    print("\nTop positive coefficients:")
    print(coef_df.head(k).to_string(index=False))
    print("\nTop negative coefficients:")
    print(coef_df.tail(k).sort_values("coef").to_string(index=False))

show_top_coefs(pipe_v3)

# ============================================================
# DECISION POINT: Moving from Linear Regression to Random Forest
# ============================================================
# Tried three linear variants:
#   1. OLS on log-price
#   2. OLS + engineered features (Mileage_per_Year)
#   3. RidgeCV regularized regression
# Best MAE ≈ 2250 € (R² ≈ 0.77). Good, but linear models assume additivity
# and miss non-linear effects and interactions (mileage impact differing
# by fuel, body, or brand segment). Next, switch to Random Forest, Decision Tree and HGBR to capture
# non-linearities and interactions without manual feature crosses.

# ================================
# MODEL 4: Random Forest Regressor
# ================================

# Reuse the same feature set as Model 1 (works well and is robust)
num_cols_rf = ["Car_Age", "Mileage_km", "Power_kW"]
cat_cols_rf = ["Fuel", "Body", "Brand_Segment"]
bin_cols_rf = ["Transmission_simple"]
X_rf = df_ml[num_cols_rf + bin_cols_rf + cat_cols_rf] 

# Random Forest Preprocess
bin_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
])

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocess_rf = ColumnTransformer([
    ("num", "passthrough", num_cols_rf),
    ("bin", bin_pipe, bin_cols_rf),
    ("cat", cat_pipe, cat_cols_rf),
])

# Use log-price target as with linear models 
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split( X_rf, y, test_size=0.2, random_state=42 )

# Define model and pipeline
rf = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
pipe_rf = Pipeline([("prep", preprocess_rf), ("model", rf)])

# Search space
param_dist = {
    "model__n_estimators": randint(400, 1000),
    "model__max_depth": [None, 10, 14, 18, 22, 26],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7]
}

search = RandomizedSearchCV(
    pipe_rf,
    param_distributions=param_dist,
    n_iter=25,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=4,
    pre_dispatch="2*n_jobs",
    verbose=0
)
search.fit(X_train_rf, y_train_rf)

best_rf = search.best_estimator_
y_pred_log_rf = best_rf.predict(X_test_rf)
y_pred_price_rf = np.expm1(y_pred_log_rf)
mae_rf = mean_absolute_error(np.expm1(y_test_rf), y_pred_price_rf)
r2_rf  = r2_score(np.expm1(y_test_rf), y_pred_price_rf)
print("Model 4: Random Forest (log target)")
print(f"RF best params: {search.best_params_}")
print(f"RF MAE: {mae_rf:.2f} €")
print(f"RF R²:  {r2_rf:.3f}")

# Permutation importance on encoded matrix
prep = best_rf.named_steps["prep"]
X_test_enc = prep.transform(X_test_rf)
if hasattr(X_test_enc, "toarray"):
    X_test_enc = X_test_enc.toarray()
est = best_rf.named_steps["model"]

perm = permutation_importance(
    est, X_test_enc, y_test_rf,
    n_repeats=5, random_state=42, n_jobs=-1,
    scoring="neg_mean_absolute_error"
)

try:
    feat_names = prep.get_feature_names_out()
except AttributeError:
    num_names = num_cols_rf
    bin_names = prep.named_transformers_["bin"].named_steps["ohe"].get_feature_names_out(bin_cols_rf)
    cat_names = prep.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(cat_cols_rf)
    feat_names = np.r_[num_names, bin_names, cat_names]

assert len(feat_names) == len(perm.importances_mean)

imp_df = (pd.DataFrame({
    "feature": feat_names,
    "perm_importance": perm.importances_mean
})
.sort_values("perm_importance", ascending=False))

print("\nTop permutation importances:")
print(imp_df.head(15).to_string(index=False))

topk = (imp_df.head(15)
        .sort_values("perm_importance"))

plt.figure(figsize=(8,6))
plt.barh(topk["feature"], topk["perm_importance"])
plt.xlabel("Permutation importance (Δ MAE in log space)")
plt.tight_layout()
plt.show()

# ===============================
# MODEL 5: DecisionTree Regressor
# ===============================

# Use the same preprocess and split as RF for fair comparison
dt = DecisionTreeRegressor(random_state=42)
pipe_dt = Pipeline([("prep", preprocess_rf), ("model", dt)])

param_dist_dt = {
    "model__max_depth": [6, 8, 10, 12, 14, 16, 18],
    "model__min_samples_leaf": [20, 50, 100, 150, 200],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3, 3e-3]
}

search_dt = RandomizedSearchCV(
    pipe_dt, 
    param_dist_dt, 
    n_iter=30, 
    scoring="neg_mean_absolute_error",
    cv=3, 
    random_state=42, 
    n_jobs=4,
    pre_dispatch="2*n_jobs",
    verbose=0
)
search_dt.fit(X_train_rf, y_train_rf)

# Extract best model and evaluate
best_dt = search_dt.best_estimator_
pred_dt = np.expm1(best_dt.predict(X_test_rf))
print("Model 5: DecisionTree Regressor")
print("DT MAE €:", mean_absolute_error(np.expm1(y_test_rf), pred_dt))
print("DT R²  :", r2_score(np.expm1(y_test_rf), pred_dt))
print("DT best params:", search_dt.best_params_)


# # =========================================
# # MODEL 6: Hist Gradient Boosting Regressor
# # =========================================

# Custom OneHotEncoder wrapper to ensure dense output across sklearn versions
def DenseOHE(**kwargs):
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

# Use the same preprocess and split as RF for fair comparison
bin_pipe_hgb = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", DenseOHE(drop="if_binary", handle_unknown="ignore"))
])

cat_pipe_hgb = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe", DenseOHE(drop="first", handle_unknown="ignore"))
])

preprocess_hgb = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ["Car_Age","Mileage_km","Power_kW"]),
        ("bin", bin_pipe_hgb, ["Transmission_simple"]),
        ("cat", cat_pipe_hgb, ["Fuel","Body","Brand_Segment"]),
    ],
    sparse_threshold=0.0
)

hgb = HistGradientBoostingRegressor(
    loss="squared_error",
    early_stopping=True,
    random_state=42
)

pipe_hgb = Pipeline([("prep", preprocess_hgb), ("model", hgb)])

param_dist = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": randint(400, 1600),
    "model__max_leaf_nodes": randint(31, 512),
    "model__max_depth": [None, 6, 8, 10, 12],
    "model__min_samples_leaf": randint(10, 200),
    "model__l2_regularization": loguniform(1e-6, 1.0),
    "model__max_bins": [255],
}
search_hgb = RandomizedSearchCV(
    pipe_hgb,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=4,
    pre_dispatch="2*n_jobs",
    verbose=0
)
search_hgb.fit(X_train_rf, y_train_rf)

# Extract best model and evaluate
best_hgb = search_hgb.best_estimator_
y_pred_log = best_hgb.predict(X_test_rf)
y_pred = np.expm1(y_pred_log)
print("Model 6: Hist Gradient Boosting Regressor")
print("HGB MAE €:", mean_absolute_error(np.expm1(y_test_rf), y_pred))
print("HGB R²  :", r2_score(np.expm1(y_test_rf), y_pred))
print("HGB best params:", search_hgb.best_params_)

# ===============================================
# Baseline and Relative MAE (rMAE) for RF and HGB
# ===============================================

from sklearn.metrics import mean_absolute_error, r2_score

# True prices (back from log)
y_true = np.expm1(y_test_rf)
y_train_prices = np.expm1(y_train_rf)
median_price = np.median(y_train_prices)

# Baseline: predict median(train)
baseline_pred = np.full_like(y_true, median_price, dtype=float)
mae_base = mean_absolute_error(y_true, baseline_pred)
r2_base  = r2_score(y_true, baseline_pred)
rmae_base = mae_base / median_price

# Helper to compute rMAE of a pipeline that predicts log-price
def rmae(model, X, y_true, median_price):
    preds = np.expm1(model.predict(X))  # model predicts log-price
    return mean_absolute_error(y_true, preds) / median_price

rmae_rf  = rmae(best_rf,  X_test_rf, y_true, median_price)
rmae_hgb = rmae(best_hgb, X_test_rf, y_true, median_price)

print(f"Baseline → MAE: {mae_base:.0f} €, R²: {r2_base:.3f}, rMAE: {100*rmae_base:.1f}%")
print(f"RF   rMAE: {100*rmae_rf:.1f}%")
print(f"HGB  rMAE: {100*rmae_hgb:.1f}%")

# ===============================
# Summary of all models
# ===============================
results = []
results += [{"model":"Linear", "MAE": mae, "R2": r2}]
results += [{"model":"Linear+MPY", "MAE": mae_v2, "R2": r2_v2}]
results += [{"model":"Ridge", "MAE": mae_v3, "R2": r2_v3}]
results += [{"model":"RF", "MAE": mae_rf, "R2": r2_rf}]
results += [{"model":"DT", "MAE": mean_absolute_error(np.expm1(y_test_rf), pred_dt),
             "R2": r2_score(np.expm1(y_test_rf), pred_dt)}]
results += [{"model":"HGB", "MAE": mean_absolute_error(np.expm1(y_test_rf), y_pred),
             "R2": r2_score(np.expm1(y_test_rf), y_pred)}]
print(pd.DataFrame(results).sort_values("MAE"))
