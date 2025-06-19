# Time Series Analysis & Forecasting

Comprehensive guide to time series analysis, forecasting methods, and deep learning approaches for temporal data.

**Last Updated:** 2025-06-19

## Table of Contents
- [Introduction](#introduction)
- [Classical Methods](#classical-methods)
- [Machine Learning Approaches](#machine-learning-approaches)
- [Deep Learning Models](#deep-learning-models)
- [Tools & Libraries](#tools--libraries)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Best Practices](#best-practices)
- [Resources](#resources)

## Introduction

Time series analysis deals with data points indexed in time order. Applications include:
- Stock price prediction
- Weather forecasting
- Energy demand prediction
- Sales forecasting
- Anomaly detection
- Healthcare monitoring

### Key Concepts
- **Trend**: Long-term direction
- **Seasonality**: Regular patterns
- **Stationarity**: Statistical properties constant over time
- **Autocorrelation**: Correlation with past values

## Classical Methods

### ARIMA Models
**[ARIMA](https://otexts.com/fpp3/arima.html)** - AutoRegressive Integrated Moving Average
- 🟢 Well-established theory
- Box-Jenkins methodology
- Suitable for univariate series

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=10)
```

### Exponential Smoothing
**[Holt-Winters](https://otexts.com/fpp3/holt-winters.html)** - Triple exponential smoothing
- Level, trend, and seasonal components
- Simple and effective
- Good for business forecasting

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit model
model = ExponentialSmoothing(
    data,
    seasonal_periods=12,
    trend='add',
    seasonal='add'
)
fit = model.fit()
```

### Prophet
**[Facebook Prophet](https://facebook.github.io/prophet/)** - Automatic forecasting
- 🆓 Open source
- 🟢 User-friendly
- Handles missing data
- Holiday effects

```python
from prophet import Prophet

# Prepare data
df = pd.DataFrame({
    'ds': dates,
    'y': values
})

# Fit and forecast
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
```

## Machine Learning Approaches

### XGBoost for Time Series
**[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting
- Feature engineering crucial
- Lag features
- Rolling statistics
- Time-based features

```python
import xgboost as xgb

# Create lag features
for lag in [1, 7, 30]:
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Rolling features
df['rolling_mean_7'] = df['value'].rolling(7).mean()
df['rolling_std_7'] = df['value'].rolling(7).std()

# Train model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
```

### Random Forest
**[scikit-learn RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)** - Ensemble method
- Robust to outliers
- Feature importance
- No extrapolation

### LightGBM
**[LightGBM](https://lightgbm.readthedocs.io/)** - Fast gradient boosting
- 🆓 Microsoft's solution
- Categorical features
- GPU acceleration
- Memory efficient

## Deep Learning Models

### LSTM (Long Short-Term Memory)
**[LSTM Networks](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)** - RNN variant
- Long-term dependencies
- Gradient vanishing solution
- Bidirectional variants

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
```

### GRU (Gated Recurrent Unit)
**[GRU Networks](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)** - Simplified LSTM
- Fewer parameters
- Faster training
- Similar performance

### Transformer Models
**[Temporal Fusion Transformer](https://github.com/jdb78/pytorch-forecasting)** - State-of-the-art
- 🔴 Advanced
- Attention mechanisms
- Multi-horizon forecasting
- Interpretable

```python
from pytorch_forecasting import TemporalFusionTransformer

# Define model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4,
)
```

### N-BEATS
**[N-BEATS](https://github.com/philipperemy/n-beats)** - Neural basis expansion
- No time series specific components
- Interpretable architecture
- Stack of fully connected layers

### DeepAR
**[Amazon DeepAR](https://docs.aws.amazon.com/sagemaker/latest/dg/deepar.html)** - Probabilistic forecasting
- 💰 AWS SageMaker
- RNN-based
- Multiple time series
- Uncertainty estimation

## Tools & Libraries

### Python Libraries

#### Core Libraries
- **[pandas](https://pandas.pydata.org/)** - Data manipulation
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[statsmodels](https://www.statsmodels.org/)** - Statistical models

#### Specialized Libraries
- **[Darts](https://github.com/unit8co/darts)** - 🌟 Unified forecasting API
  ```python
  from darts import TimeSeries
  from darts.models import TCNModel
  
  # Load data
  series = TimeSeries.from_dataframe(df)
  
  # Train model
  model = TCNModel(
      input_chunk_length=24,
      output_chunk_length=12
  )
  model.fit(series)
  ```

- **[sktime](https://www.sktime.org/)** - scikit-learn for time series
- **[tslearn](https://tslearn.readthedocs.io/)** - ML for time series
- **[PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/)** - Deep learning forecasting

#### AutoML for Time Series
- **[AutoTS](https://github.com/winedarksea/AutoTS)** - Automatic time series
- **[FEDOT](https://github.com/nccr-itmo/FEDOT)** - AutoML framework
- **[AtsPy](https://github.com/firmai/atspy)** - Automated Python

### Visualization Tools
- **[Plotly](https://plotly.com/python/)** - Interactive plots
- **[Altair](https://altair-viz.github.io/)** - Declarative visualization
- **[TSViz](https://github.com/timeseriesAI/tsai)** - Time series specific

## Datasets & Benchmarks

### Popular Datasets

#### Financial
- **[Yahoo Finance](https://finance.yahoo.com/)** - Stock prices
- **[Quandl](https://www.quandl.com/)** - Financial data
- **[Alpha Vantage](https://www.alphavantage.co/)** - Free API

#### Energy & Weather
- **[UCI Electricity](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)** - Load diagrams
- **[NOAA Weather](https://www.ncdc.noaa.gov/)** - Climate data
- **[Solar Power](https://www.nrel.gov/grid/solar-power-data.html)** - Generation data

#### Benchmarks
- **[M Competitions](https://www.mcompetitions.unic.ac.cy/)** - Forecasting competitions
- **[Monash Repository](https://forecastingdata.org/)** - Curated datasets
- **[TimeSeriesBenchmark](https://github.com/Mcompetitions/M5-methods)** - Standard benchmarks

### Data Characteristics
| Dataset | Frequency | Length | Features | Domain |
|---------|-----------|---------|----------|---------|
| M4 Competition | Various | 100-9,000 | Univariate | Mixed |
| Electricity | Hourly | 4 years | Multivariate | Energy |
| Traffic | Hourly | 2 years | Multivariate | Transportation |
| Retail | Daily | 3 years | Hierarchical | Sales |

## Best Practices

### Data Preprocessing
1. **Missing Value Handling**
   ```python
   # Forward fill
   df.fillna(method='ffill')
   
   # Interpolation
   df.interpolate(method='linear')
   
   # Advanced imputation
   from sklearn.impute import KNNImputer
   ```

2. **Outlier Detection**
   - Isolation Forest
   - Local Outlier Factor
   - Statistical methods (IQR, Z-score)

3. **Normalization**
   - Min-Max scaling
   - Standard scaling
   - Robust scaling

### Feature Engineering
```python
# Time features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['quarter'] = df.index.quarter

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Lag features
for lag in [1, 24, 168]:  # hour, day, week
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Rolling statistics
windows = [24, 168]  # day, week
for window in windows:
    df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
    df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
```

### Model Selection
1. **Start Simple**: Naive, moving average
2. **Classical Models**: ARIMA, ETS
3. **Machine Learning**: If enough data
4. **Deep Learning**: Complex patterns, large data

### Validation Strategies
```python
# Time series split
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

# Walk-forward validation
def walk_forward_validation(data, n_test):
    predictions = []
    history = data[:-n_test]
    
    for i in range(n_test):
        model = fit_model(history)
        yhat = model.forecast()
        predictions.append(yhat)
        history.append(data[-n_test + i])
    
    return predictions
```

### Evaluation Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
- **sMAPE**: Symmetric MAPE
- **MASE**: Mean Absolute Scaled Error

## Advanced Topics

### Hierarchical Forecasting
```python
from hts import HTSRegressor

# Define hierarchy
hierarchy = {
    'total': ['A', 'B'],
    'A': ['A1', 'A2'],
    'B': ['B1', 'B2', 'B3']
}

# Fit hierarchical model
hts = HTSRegressor(model='auto', revision_method='OLS')
hts.fit(df, hierarchy)
```

### Probabilistic Forecasting
- Prediction intervals
- Quantile regression
- Monte Carlo dropout
- Ensemble uncertainty

### Online Learning
```python
from river import time_series

# Online ARIMA
model = time_series.SNARIMAX(
    p=2,
    d=1,
    q=2,
    m=12,
    sd=1
)

# Update with new data
for x, y in stream:
    y_pred = model.predict_one(x)
    model.learn_one(x, y)
```

## Resources

### Courses & Tutorials
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Free online book
- [Time Series Analysis in Python](https://www.datacamp.com/courses/time-series-analysis-in-python) - DataCamp
- [Deep Learning for Time Series](https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction) - Coursera

### Papers
- [N-BEATS: Neural basis expansion analysis](https://arxiv.org/abs/1905.10437)
- [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)
- [DeepAR: Probabilistic forecasting](https://arxiv.org/abs/1704.04110)

### Competitions
- [Kaggle Time Series](https://www.kaggle.com/competitions?hostSegmentIdFilter=5) - Various competitions
- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) - Walmart sales
- [Web Traffic Forecasting](https://www.kaggle.com/c/web-traffic-time-series-forecasting) - Wikipedia traffic

### Communities
- [r/timeseries](https://reddit.com/r/timeseries) - Reddit community
- [Forecasting Stack Exchange](https://stats.stackexchange.com/questions/tagged/forecasting) - Q&A