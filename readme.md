# Car Market Analysis (Slovakia)

## What this is
I scraped a snapshot of the Slovak used-car market and built models to predict prices.  
It’s a portfolio/learning project — my background is metallurgy (PhD) and I’m moving into Data Science.

## Data pipeline (scrape → clean)
This repo includes a simple 3-step pipeline I used to build the dataset:
1) `link_collector.py` — collects listing URLs  
2) `detail_scraper.py` — fetches details for each listing  
3) `clean_preprocess.py` — cleaning & feature prep

## Data
- One-day snapshot from **August 2025** (~20k listings after cleaning)
- Columns: Price, Brand, Model, Year, Mileage_km, Power_kW, Fuel, Transmission, etc.
- Caveat: one day ≠ the whole market. With a few weeks of data this would be more stable.

## EDA in a sentence
Older and high-mileage cars are cheaper (no surprise), power helps, premium brands carry a premium, and automatics + hybrids/EVs trend higher.

## Modeling
- **Target:** log-price (for stability), predictions converted back to €
- **Features:** `Car_Age`, `Mileage_km`, `Power_kW` + `Fuel`, `Body`, `Brand_Segment`, `Transmission_simple`
- **Encoding:** fixed category schema to keep train/test consistent

**Results (held-out test):**
- **Random Forest:** **MAE ≈ €1.83k**, **R² ≈ 0.85**  
- **HistGradientBoosting:** **MAE ≈ €1.85k**, **R² ≈ 0.85**  
- **Baseline (median):** **MAE ≈ €5.0k**

Biggest drivers: **Car_Age**, **Power_kW**, **Mileage_km**, plus **Brand_Segment** and **Body**.

## Limitations
One-day data, listing prices (not final sale), a few “unknown” categories. No seasonal or geographic effects.

## Repo usage
- EDA & modeling are in notebooks:
  - `EDA_notebook.ipynb`
  - `modeling.ipynb` (final models & metrics)
- Install deps:
  ```bash
  pip install -r requirements.txt

