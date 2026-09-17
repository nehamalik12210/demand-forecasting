# Demand Forecasting

Forecasting daily store-level sales using time series features and gradient-boosted trees, with a chronological train/test split to prevent data leakage.

## Results

| Model | MAPE (%) | RMSE | RMSPE (%) |
| --- | --- | --- | --- |
| **XGBoost (store-level)** | **10.73** | **959.33** | **14.79** |
| LightGBM (store-level) | 10.82 | 961.44 | 15.15 |
| SARIMA (aggregate-level)* | 12.87 | 1101.51 | 16.34 |
| Naive (same-day-last-week) | 37.64 | 3063.00 | 48.95 |

*SARIMA is fit on daily sales averaged across all stores, a smoother series than the store-day level the ML models predict on. Its numbers are not directly comparable to the store-level rows.

**Best model**: XGBoost with 10.73% MAPE -- predictions are off by ~11% on average, roughly 3.5x better than the naive baseline.

## Key Findings

1. **28-day rolling mean of sales** is the single strongest predictor by a wide margin -- recent sustained demand level dominates over any individual day's features
2. **Promotions** are the second-strongest driver -- actual sales lift on promo days is +38.8%; the model's own predicted lift on promo days is +50.4%
3. Among lag features, **lag_1** (yesterday's sales) is by far the most useful -- lag_7, lag_14, and lag_28 don't crack the top 15 feature importances
4. **DayOfWeek** captures real weekly seasonality but is a secondary signal once rolling averages, promotions, and lag_1 are in the model
5. **Store-level median MAPE = 10.22%** (mean 10.79%) -- most stores are well-predicted; the worst store (MAPE 81.6%) likely has an unmodeled local factor

## Project Structure

```
Demand Forecasting/
    README.md
    requirements.txt
    .gitignore
    models/
        forecasting_model.pkl                                       # Trained XGBoost model (produced by notebook 2)
    notebooks/
        eda_output_file.ipynb                                       # Exploratory Data Analysis (with outputs)
        feature_engineering_and_modeling_output_file.ipynb          # Features + Modeling (with outputs)
```

## Notebooks

### 1. EDA (`eda_output_file.ipynb`)

- Time series overview (daily, weekly, monthly patterns)
- Stationarity test (ADF)
- STL decomposition (trend + seasonal + residual)
- Promotion and holiday impact analysis
- Store type and assortment comparisons
- Competition distance effect

### 2. Features + Modeling (`feature_engineering_and_modeling_output_file.ipynb`)

- Lag features (1, 7, 14, 28 days) with leakage prevention
- Rolling window statistics (7-day, 28-day mean/std)
- Holiday proximity and competition features
- Chronological train/test split (final 6 weeks held out as a single time-based holdout)
- 4 models: Naive, SARIMA, LightGBM, XGBoost
- Store-level error analysis
- Business translation (promo lift, day-of-week patterns)

## Dataset

[Rossmann Store Sales](https://www.kaggle.com/datasets/nehamalik10/demand-forecasting-dataset) -- 1,115 stores, 2.5 years of daily sales data.

## Tech Stack

Python, pandas, LightGBM, XGBoost, statsmodels (ADF test, STL, SARIMA), matplotlib, seaborn

## How to Run

Upload the notebooks to [Kaggle](https://www.kaggle.com) with the dataset linked above, and run cells in order. Each notebook is self-contained: notebook 2 loads and re-engineers features from the raw `train.csv`/`store.csv` files itself rather than depending on saved output from notebook 1.
