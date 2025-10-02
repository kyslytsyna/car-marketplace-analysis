# Car Market Analysis (Slovakia)

## What this is
I scraped a snapshot of the Slovak used-car market and built some models to predict prices.  
It’s a portfolio/learning project — my background is metallurgy (PhD), and I’m moving into Data Science.

## Data
- One-day snapshot from August 2025 (~20k listings after cleaning)
- Columns like: Price, Brand, Model, Year, Mileage_km, Power_kW, Fuel, Transmission, etc.
- Caveat: one day ≠ the whole market. With a few weeks of data this would be more stable.

## EDA in a sentence
Older and high-mileage cars are cheaper (no surprise), power helps, premium brands carry a premium, and automatics + hybrids/EVs trend higher.

## Modeling (v1)
- Target: **log-price** (for stability), then convert back to €
- Features: `Car_Age`, `Mileage_km`, `Power_kW` + `Fuel`, `Body`, `Brand_Segment`, `Transmission_simple`
- Categorical encoding uses a fixed schema to stay consistent across train/test

**Results (held-out test):**
- **Random Forest:** **MAE ~ €1.83k**, **R² ~ 0.85**
- **HistGradientBoosting:** **MAE ~ €1.85k**, **R² ~ 0.85**
- **Baseline (median):** **MAE ~ €5.0k**

So the tree models clearly beat the baseline. Biggest drivers: **Car_Age**, **Power_kW**, **Mileage_km**, plus brand segment and body type.

## Limitations
One-day data, listing prices (not final sale), a few “unknown” categories. No seasonal/geography effects.

## What’s next
I’ll keep tuning the tree models (and maybe try a tiny ensemble) to chip away at MAE.

## Run it
```bash
pip install -r requirements.txt
# EDA
jupyter notebook EDA_notebook.ipynb
# Train models via script (trains 6 models end-to-end and prints metrics)
python ML.py
