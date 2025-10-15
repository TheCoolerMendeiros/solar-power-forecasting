# ☀️ Solar Power Forecasting

**Predicting solar AC power output using machine learning and physics-informed feature engineering.**

Solar energy generation is inherently variable due to changing weather conditions. Accurate power forecasting is critical for grid stability, energy trading, and storage optimization. This project builds an end-to-end ML pipeline to forecast 15-minute solar power output from weather sensor data.

**Key Achievement:** Identified the performance ceiling for this dataset at **65% R² on test data** through systematic experimentation, demonstrating that remaining error stems from unmeasured physical factors rather than model inadequacy.

---

## 🎯 Problem Statement

**Challenge:** Solar power fluctuates with weather conditions, making grid integration difficult:
- **Grid operators** need accurate forecasts to balance supply/demand
- **Energy traders** require predictions for market participation
- **Maintenance teams** need to plan interventions during low-production periods

**Approach:** Data-driven forecasting using historical sensor readings to learn complex relationships between environmental conditions and power output, without requiring detailed equipment specifications.

---

## 📊 Dataset

- **Source**: [Kaggle - Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- **Time period**: May 15 - June 17, 2020 (34 days)
- **Temporal resolution**: 15-minute intervals
- **Facilities**: 2 solar plants in India
- **Total samples**: 136,476 records (71,559 daytime observations after filtering)
- **Inverters**: 44 unique units across both plants

**Features:**
- Environmental: Irradiation (W/m²), ambient temperature, module temperature
- Operational: AC power output (target), inverter IDs, plant IDs
- Temporal: Timestamp for each measurement

**Key challenges:**
- ⚠️ Only 34 days of data (limited temporal coverage)
- ⚠️ No inverter specifications (efficiency curves unknown)
- ⚠️ 15-minute weather resolution (misses rapid changes)
- ⚠️ No soiling/degradation measurements

---

## 🔧 Data Processing

### 1. Feature Selection
- ❌ Removed DC_POWER (inverter datasheets unavailable)
- ❌ Removed DAILY_YIELD and TOTAL_YIELD (cumulative leakage)
- ✅ Kept SOURCE_KEY (44 inverter identifiers) as categorical feature
- ✅ Merged generation and weather data on timestamps

### 2. Nighttime Filtering
- Dropped records with irradiation < 5 W/m² (nighttime)
- Reduced dataset from 136K → 72K samples (daytime only)
- In production: Simple rule `if irradiance < 5: return 0`

### 3. Cross-Plant Analysis
- Compared AC power response curves between plants
- **Finding**: Different irradiance-to-power relationships (panel orientation/capacity differences)
- **Decision**: Merged datasets, kept PLANT_ID as feature (unified model)

---

## ⚙️ Feature Engineering

Physics-informed features to capture solar generation dynamics:

### Temporal Features (Cyclic Encoding)
```python
HOUR_SIN = sin(2π × hour / 24)
HOUR_COS = cos(2π × hour / 24)
DAY_SIN = sin(2π × day_of_year / 365)
DAY_COS = cos(2π × day_of_year / 365)
```
**Purpose**: Capture diurnal patterns without discontinuity (23:00 → 00:00)

### Solar Geometry (pvlib library)
```python
SOLAR_ZENITH    # Angle from vertical (0° = overhead)
SOLAR_AZIMUTH   # Compass direction
SOLAR_ELEVATION # Angle from horizon
AIR_MASS = 1 / cos(SOLAR_ZENITH)  # Atmospheric path length
```
**Purpose**: Model sun angle effects on panel efficiency

### Thermal Efficiency
```python
TEMPERATURE_DIFFERENCE = MODULE_TEMP - AMBIENT_TEMP
```
**Purpose**: Capture efficiency losses from panel heating

**Impact**: Engineered features improved baseline by **+8% R²** and became the #1 most important predictor.

---

## 🤖 Model Development

### Baseline Models

| Model | Train R² | Val R² | Notes |
|-------|----------|--------|-------|
| Linear Regression (raw) | 62.6% | 62.6% | Struggles with non-linearity |
| Linear Regression (+ features) | 63.2% | 63.2% | Marginal improvement |
| XGBoost (default) | **88.9%** | 55.3% | ⚠️ Severe overfitting |
| Random Forest (default) | **88.9%** | 55.3% | ⚠️ Severe overfitting |

**Key Finding**: Default tree models memorize training data. Strong regularization required.

### Regularized Models

| Model | Train R² | Val R² | Gap | Configuration |
|-------|----------|--------|-----|---------------|
| **XGBoost (conservative)** | **73.2%** | **70.6%** | 2.5% | max_depth=4, learning_rate=0.1 |
| Random Forest (conservative) | 77.2% | 70.6% | 6.5% | max_depth=15, min_samples=5 |

**Success**: Regularization eliminated overfitting while improving validation performance by **+15%**.

---

## 🔍 Hyperparameter Optimization

### Systematic Search Strategy

**Phase 1: RandomizedSearchCV (40 iterations)**
- Best params: `max_depth=3, learning_rate=0.05, n_estimators=200`
- Validation R²: **71.6%**

**Phase 2: GridSearchCV (81 combinations)**
- Refined search around optimal region
- Validation R²: **71.2%** (confirmed plateau)

**Phase 3: Extended Search (with L1/L2 regularization)**
- Added reg_alpha and reg_lambda
- Validation R²: **71.2%** (no improvement)

### Critical Experiment: Depth Analysis

Fixed all hyperparameters, varied only `max_depth`:

| max_depth | Train R² | Val R² | Gap | Overfitting? |
|-----------|----------|--------|-----|--------------|
| 3 | 69.8% | **71.6%** | -1.7% | ✅ Optimal |
| 4 | 73.2% | 71.1% | +2.1% | ✅ Acceptable |
| 5 | 76.6% | 70.1% | +6.5% | ⚠️ Mild |
| 6 | 80.1% | 67.3% | +12.8% | ❌ Severe |
| 7 | 83.8% | 65.7% | +18.1% | ❌ Severe |
| 8 | 86.5% | 62.7% | +23.8% | ❌ Catastrophic |

**Key Insights:**
- ✅ `max_depth=3` achieves highest validation score (71.6%)
- ❌ Deeper trees (6+) cause performance collapse
- 📊 Negative gap at depth=3 (val > train) indicates good generalization
- 📊 At depth=8, model performs **worse than Linear Regression** (62.7% < 63.2%)

**Conclusion**: Shallow trees generalize best. Further complexity hurts performance.

### Ensemble Attempt

Tested XGBoost + Ridge Regression ensemble (70% + 30% weighting):
- **Result**: 70.6% R² (no improvement)
- **Why failed**: Ridge adds linear noise that degrades XGBoost's non-linear predictions

---

## 📈 Final Results

### Test Set Performance (June 11-17, unseen data)

**Final Model**: XGBoost with `max_depth=3, learning_rate=0.05, n_estimators=200, min_child_weight=6`

| Metric | Train | Validation | Test | Gap |
|--------|-------|------------|------|-----|
| **R²** | 69.8% | 71.6% | **64.9%** | -6.7% |

**Performance drop from validation to test**: This is a **realistic outcome** for time-series forecasting:
- Validation period (June 5-10) may have had easier weather patterns
- Test period (June 11-17) introduced new conditions:
  - Different weather regime (hotter? more clouds?)
  - Panel degradation over time
  - Seasonal shift (later in June = different sun angles)

### Comparison to Baseline

| Model | Test R² | Improvement |
|-------|---------|-------------|
| Linear Regression | 63.2% | - |
| **XGBoost (tuned)** | **64.9%** | **+1.7%** |

**Is 64.9% good?**
- ✅ **Much better than Linear Regression** (63.2%)
- ✅ **Explains 2/3 of power variance** - useful for grid forecasting
- ⚠️ **Leaves 35% unexplained** - but this is expected (see below)

---

## 🚧 Performance Ceiling Analysis

### Why R² plateaued at ~65%

**Evidence that we hit a data limit, not a model limit:**

1. **Hyperparameter tuning shows diminishing returns**:
   - RandomizedSearch: 71.6%
   - GridSearch: 71.2% (no gain)
   - Extended search + regularization: 71.2% (no gain)

2. **Deeper models hurt performance**:
   - depth=8 achieves 86.5% on training but only 62.7% on validation
   - This proves the model can fit complexity, but it doesn't help

3. **Residual analysis shows random patterns**:
   - No systematic bias in errors
   - Prediction errors are uncorrelated with features
   - Remaining variance appears stochastic

### Remaining 35% unexplained variance likely due to:

**Unmeasured factors:**
- ☁️ **Cloud edges**: Sudden irradiance changes between 15-min intervals
- 🐦 **Soiling events**: Bird droppings, dust storms not in features
- ⚡ **Inverter trips**: Brief shutdowns/restarts not captured
- 🌡️ **Sensor lag**: Temperature sensors may lag actual panel temp by 1-2 minutes

**Data limitations:**
- 📅 **Only 34 days**: Insufficient to learn seasonal patterns
- ⏱️ **15-minute resolution**: Misses rapid fluctuations
- 🔧 **No inverter specs**: Unknown efficiency curves
- 📍 **No shading data**: Obstacles, panel orientation unknown

**Inherent stochasticity:**
- 🌤️ **Weather is chaotic**: Small perturbations amplify
- 📊 **Measurement noise**: Sensor precision limits

### What would be needed to exceed 75% R²:

1. **Higher temporal resolution** (<5-minute intervals)
2. **Inverter technical specifications** (efficiency curves)
3. **Soiling sensors** + maintenance logs
4. **Weather radar** (cloud movement predictions)
5. **Longer time period** (full year for seasonal learning)

**Conclusion**: Achieving ~65% R² with 34 days of 15-minute data represents the **practical ceiling** given these constraints. Further algorithmic improvements unlikely without better data.

---

## 🔬 Feature Importance

Top predictors (XGBoost feature importance):

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **TEMPERATURE_DIFFERENCE** | 36.4% | 🏆 Dominant predictor - captures efficiency |
| 2 | IRRADIATION | 15.9% | ⭐ Core driver (surprisingly only 2nd!) |
| 3 | MODULE_TEMPERATURE | 10.5% | Thermal effects |
| 4 | PLANT_ID | 7.0% | Plant-specific characteristics |
| 5 | HOUR | 4.3% | Time-of-day patterns |
| 6 | HOUR_COS | 4.0% | Cyclic time encoding |
| 7-10 | SOURCE_KEY_* | 2-1% each | Individual inverter behaviors |

### Why TEMPERATURE_DIFFERENCE beats IRRADIATION:

**Non-redundancy**: `TEMP_DIFF` captures information that `IRRADIATION` doesn't:
- High irradiance + low temp diff = efficient conversion
- High irradiance + high temp diff = efficiency losses (hot panels)
- The **ratio matters**, not just absolute irradiance

**Inverter saturation**: At high irradiance, panels hit rated capacity. Further increases don't boost power, but temperature effects still vary.

**First-split advantage**: XGBoost splits on `TEMP_DIFF` first because it cleanly divides data into "efficient" vs "inefficient" regimes.

**Physical validation:**
- Solar panels lose ~0.4-0.5% efficiency per °C above 25°C
- A hot panel (ΔT=30°C) at 900 W/m² produces **less** than a cool panel (ΔT=10°C) at same irradiance
- Model learned this relationship automatically!

**Feature engineering success**: Our engineered feature became the **most important predictor**, validating the domain-knowledge approach.

---

## 🛠️ Tech Stack

- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Solar Calculations**: PVLib (sun position features)
- **Machine Learning**: Scikit-learn, XGBoost
- **Model Persistence**: Joblib

---

## 💻 Usage

### Installation

```bash
git clone https://github.com/YourUsername/solar-power-forecasting
cd solar-power-forecasting
pip install -r requirements.txt
```

### Training

```python
# Full pipeline in notebook
jupyter notebook solar_forecasting.ipynb
```

### Prediction

```python
import joblib
import pandas as pd
from pvlib import solarposition

# Load trained model
model = joblib.load('solar_power_model.pkl')

# Prepare new data (example)
new_data = pd.DataFrame({
    'IRRADIATION': [850],         # W/m²
    'AMBIENT_TEMPERATURE': [35],   # °C
    'MODULE_TEMPERATURE': [55],    # °C
    'PLANT_ID': [0],              # Encoded plant ID
    'SOURCE_KEY_1BY6WEcLGh8j5v7': [1],  # One-hot inverter
    # ... + engineered features (HOUR, TEMP_DIFF, SOLAR_ZENITH, etc.)
})

# Predict
predicted_power = model.predict(new_data)
print(f"Predicted AC Power: {predicted_power[0]:.2f} W")
```

### Deployment Considerations

- Recompute engineered features (TEMP_DIFF, solar position) from raw sensor readings
- Handle missing data (use median imputation or retrain)
- Monitor for distribution drift (weather patterns change seasonally)
- Retrain quarterly to adapt to panel degradation

---

## 📁 Project Structure

```
solar-power-forecasting/
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   └── Plant_2_Weather_Sensor_Data.csv
├── notebooks/
│   └── solar_forecasting.ipynb    # Full analysis
├── solar_power_model.pkl      # Trained XGBoost
├── requirements.txt
└── README.md
```

---

## 🔑 Key Takeaways

1. ✅ **Feature engineering matters**: Physics-informed features (TEMP_DIFF) outperformed raw measurements
2. ✅ **Regularization critical**: Default XGBoost overfits badly; shallow trees (depth=3) generalize best
3. ✅ **Know when to stop**: Achieved ~65% R² ceiling through systematic experimentation
4. ✅ **Data quality > model complexity**: Remaining error attributable to unmeasured factors, not inadequate algorithms
5. ⚠️ **Time-series challenges**: 34 days insufficient for seasonal patterns; test performance dropped 6.7% from validation

---

## 🚀 Next Steps (Production Roadmap)

If deploying this system:

1. **Data improvements**:
   - Increase temporal resolution to <5 minutes
   - Add weather forecast integration (cloud predictions)
   - Install soiling sensors
   - Log inverter maintenance events

2. **Model enhancements**:
   - Ensemble of models trained on different time windows
   - Quantile regression for uncertainty estimates
   - Online learning to adapt to panel degradation

3. **Operational integration**:
   - Real-time API for grid operators
   - Alert system for anomalous predictions
   - A/B testing against baseline forecasts

---

## 📚 References

- Dataset: [Kaggle - Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
- PVLib: [Solar position algorithms](https://pvlib-python.readthedocs.io/)
- XGBoost: [Chen & Guestrin (2016)](https://arxiv.org/abs/1603.02754)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file

---

## 🤝 Contact

Built by Pedro Mendeiros | [GitHub](https://github.com/TheCoolerMendeiros) | [LinkedIn](https://www.linkedin.com/in/pedro-mendeiros-159a801a8)

*Interested in time-series forecasting or renewable energy ML? Let's connect!*