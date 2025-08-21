# Car Market Analysis in Slovakia

## Project overview
This project is about analyzing the used car market in Slovakia.  
I collected the data myself from autobazar.sk and then explored how car prices are connected with age, mileage, brand, fuel, etc. The main idea was to understand depreciation and price patterns, and later to try some machine learning for price prediction.  

This is more like a learning project, I have background in metallurgy (PhD) but I want to move into Data Science, so I am building portfolio step by step.

---

## Dataset
- Source: scraped listings from autobazar.sk  
- Data: one day snapshot (August 2025)  
- Size: ~20 000 listings after cleaning  
- Main columns: Price, Brand, Model, Year, Mileage, Power, Fuel, Transmission, Condition, Emission, Engine volume  

Important: since the dataset was collected only for one day, it might not be fully representative of the whole market. If I had collected it for longer (like once a week for a month or two), the results would be more stable.

---

## Exploratory Data Analysis (EDA)

### 1. Distributions
- Price, Mileage, Power, Year checked with histograms and log-transform.
- Trimmed views were used to avoid extreme outliers.

### 2. Categories
- Brand: premium brands like BMW, Mercedes, Audi are always higher, while Škoda, VW, Ford are more mixed.  
- Fuel: electric and hybrid are most expensive, LPG and CNG the cheapest.  
- Transmission: automatic cars generally higher priced.  
- Condition: new/demo much more expensive than used, damaged cars are at the bottom.  
- Emission: newer emission standards (Euro 6) = higher price.  
- Engine volume: electric cars (0 L) are on top, then generally bigger engine means higher price.  

### 3. Price vs Year
- Older cars are cheaper, but there are exceptions (classic cars from 60s–70s are expensive).  
- New cars after 2010 are progressively more expensive.  
- Premium brands show stronger value retention.  

### 4. Price vs Mileage
- Clear depreciation with higher mileage.  
- After 200k km the drop slows down and prices just stabilize low.  

### 5. Depreciation insights
- Mercedes and BMW hold value longer than others.  
- Big cliffs: 50–100k km and 150–200k km, where prices fall faster.  
- Economy brands are more sensitive to mileage.  

### 6. Outliers
- Simple IQR flagged a lot of cars, but contextual (brand+mileage) filtering shows ~530 suspicious cases.  
- High outliers: overpriced SUVs with heavy mileage.  
- Low outliers: very cheap small cars (sometimes maybe mistakes or distressed sales).  
- There are horizontal “lines” in the plots at 10k, 15k, etc. → sellers like round numbers.

### 7. Relationships
- Age has the strongest negative correlation with price (-0.7).  
- Mileage also negative (-0.36) but weaker than age.  
- Power has positive correlation (+0.47).  
- Cross-tabs: automatic cars more expensive across almost all brands. Electric/hybrids show consistent premium.  

---

## Limitations
- Data collected only one day → snapshot, not the full picture.  
- Some noisy or missing values (models listed as “unknown”, strange engine data).  
- Prices are listing prices, not actual sale prices.  

---

## Next steps (ML)
- Prepare features (categorical encoding, combinations like Age×Mileage).  
- Try simple models: linear regression, decision trees.  
- Try advanced: random forest, gradient boosting.  
- Measure performance (RMSE, MAE, R²).  
- Maybe deploy small app for price prediction later.  

---

## How to run
1. Clone repo  
2. Install requirements  
    bash
    pip install -r requirements.txt
3. Run EDA notebook
    jupyter notebook notebooks/eda_car_market.ipynb