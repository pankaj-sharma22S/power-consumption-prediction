# ⚡ Electricity Consumption Forecasting using LSTM

<p align="center">

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Google Colab"/>

</p>

<p align="center">
  <b>Predict the next hour's electricity consumption using historical energy data and deep learning.</b>
</p>

---

## 🌟 Overview

Electricity consumption changes throughout the day depending on usage patterns and other factors. Forecasting future consumption can help people and organizations better understand their energy usage and plan ahead.

This project uses a **Long Short-Term Memory (LSTM)** neural network to forecast hourly household electricity consumption using historical electricity readings.

The model learns temporal patterns from previous hourly consumption values and predicts future energy consumption in kilowatt-hours (kWh).

### 🎯 Project Goals

- 📊 Analyze and preprocess real-world electricity consumption data.
- 🧹 Handle missing measurements and incomplete hourly periods.
- 🕒 Convert minute-level readings into hourly energy consumption.
- 🧠 Train an LSTM model for time-series forecasting.
- 📈 Evaluate model performance using regression metrics.
- 🔮 Predict electricity consumption for multiple future hours.
- 💾 Save and reload the trained model for reuse.

## ✨ Features

| Feature | Description |
|---|---|
| 🧹 Data preprocessing | Handle missing values and prepare timestamps |
| 🕒 Hourly aggregation | Convert minute-level readings into hourly consumption |
| 📅 Time-based splitting | Divide data chronologically into training, validation, and testing sets |
| ⚖️ Robust scaling | Scale data using statistics learned from the training set |
| 🧠 Deep learning | Use a stacked LSTM neural network |
| ⏳ Time-series sequences | Use the previous 24 hours to predict the next hour |
| 📉 Model evaluation | Calculate MAE, RMSE, and R² |
| 🔮 Multi-hour forecasting | Generate sequential predictions for a user-selected number of hours |
| 💾 Model persistence | Save and reload the trained model and scaler |

## 🛠️ Tech Stack

<p>
  <img src="https://img.shields.io/badge/Python-Language-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-Data_Processing-purple?style=flat-square&logo=pandas" alt="Pandas"/>
  <img src="https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=flat-square&logo=numpy" alt="NumPy"/>
  <img src="https://img.shields.io/badge/TensorFlow-Deep_Learning-FF6F00?style=flat-square&logo=tensorflow" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Preprocessing_&_Metrics-F7931E?style=flat-square&logo=scikitlearn" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square" alt="Matplotlib"/>
  <img src="https://img.shields.io/badge/Seaborn-Statistical_Visualization-76B900?style=flat-square" alt="Seaborn"/>
  <img src="https://img.shields.io/badge/Google_Colab-Development_Environment-F9AB00?style=flat-square&logo=googlecolab" alt="Google Colab"/>
</p>

### Technologies and Their Roles

- **Python:** Main programming language.
- **Pandas:** Data cleaning, timestamp handling, and hourly resampling.
- **NumPy:** Numerical operations and sequence preparation.
- **TensorFlow/Keras:** Building, training, and running the LSTM model.
- **Scikit-learn:** RobustScaler and evaluation metrics.
- **Matplotlib/Seaborn:** Visualizing model performance and forecasting results.
- **Google Colab:** Cloud-based notebook environment for experimentation and training.

## 📂 Dataset

**Dataset:** Individual Household Electric Power Consumption

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains household electricity measurements collected at approximately one-minute intervals from December 2006 to November 2010.

Important columns in the original dataset include:

- `Date` and `Time`
- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

The project uses `Global_active_power` to calculate hourly energy consumption.

### ⚡ Target Variable

The target variable is:

`Hourly_Energy_kWh`

For each complete hour, the minute-level active power readings are summed and divided by 60 to estimate hourly energy consumption in kWh.

Only hours with all 60 minute-level measurements are retained.

## 🔄 Project Workflow

```text
Raw Electricity Dataset
          |
          v
Data Cleaning and Missing-Value Handling
          |
          v
Datetime Creation and Chronological Sorting
          |
          v
Hourly Energy Aggregation
          |
          v
Train / Validation / Test Split
          |
          v
Robust Scaling
          |
          v
24-Hour Sequence Creation
          |
          v
LSTM Model Training
          |
          v
Model Evaluation
          |
          v
Future Electricity Forecasting
```

### 1. Data Preprocessing

- Loaded the dataset using Pandas.
- Converted date and time columns into a datetime feature.
- Handled missing measurements.
- Sorted readings chronologically.
- Aggregated minute-level power readings into hourly energy consumption.
- Excluded incomplete hours to maintain consistent aggregation.

### 2. Time-Series Splitting

The dataset was split chronologically to prevent future observations from leaking into the training data.

| Dataset | Percentage |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The scaler was fitted only on the training data and then applied to the validation and test data.

### 3. Sequence Creation

The model uses a lookback window of 24 hours.

**Input:** Electricity consumption from the previous 24 consecutive hours.

**Output:** Predicted electricity consumption for the next hour.

The LSTM input shape is:

`(samples, 24, 1)`

- `samples`: Number of training examples.
- `24`: Previous 24 hourly readings.
- `1`: One input feature, hourly energy consumption.

Only sequences with consecutive hourly timestamps are retained.

## 🧠 LSTM Model Architecture

The project uses a stacked LSTM architecture to learn temporal relationships in electricity consumption.

| Layer | Configuration | Purpose |
|---|---|---|
| LSTM 1 | 64 units | Learn temporal patterns |
| Dropout | 20% | Help reduce overfitting |
| LSTM 2 | 32 units | Learn higher-level sequential patterns |
| Dense | 16 neurons, ReLU | Learn nonlinear relationships |
| Dense Output | 1 neuron | Predict next-hour consumption |

**Total trainable parameters:** 29,857

### ⚙️ Training Configuration

- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)
- Evaluation metrics: MAE and RMSE
- Maximum epochs: 50
- Batch size: 32
- Early stopping: Patience of 5 epochs
- Best validation weights: Restored using early stopping

The model is trained with chronological sequences and `shuffle=False`.

## 📊 Model Performance

The following results were obtained during evaluation on the test set.

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.2863 kWh |
| Root Mean Squared Error (RMSE) | 0.3988 kWh |
| R² Score | 0.5062 |

### What Do These Results Mean?

- **MAE — 0.2863 kWh:** The average absolute difference between predicted and actual hourly consumption is approximately 0.286 kWh.
- **RMSE — 0.3988 kWh:** Larger prediction errors receive more weight, making RMSE useful for assessing substantial mistakes.
- **R² — 0.5062:** The model explains approximately 50.6% of the variation in the test targets relative to a mean-prediction baseline.

These metrics describe regression performance; they do not mean the model has 50.6% prediction accuracy.

## 🔮 Multi-Hour Forecasting

After training, the model can predict electricity consumption for multiple future hours.

For example, the user can request forecasts for the next 1, 6, 12, or 24 hours.

The forecasting process works as follows:

1. Select the latest 24 historical hourly readings.
2. Scale the readings using the saved scaler.
3. Predict the next hour using the trained LSTM.
4. Convert the prediction back into kWh.
5. Add the scaled prediction to the input history.
6. Repeat the process for the requested forecast horizon.
7. Display the predicted date, time, and consumption.

This is known as **recursive or autoregressive multi-step forecasting** because later predictions depend on earlier predictions.

> Note: Forecasts farther into the future can become less reliable as prediction errors accumulate. The final historical window must also contain 24 consecutive hourly observations.

## 💾 Saving and Loading the Model

The trained model and preprocessing scaler can be saved for reuse.

```python
model.save("electricity_lstm.keras")
```

Save the scaler separately:

```python
import joblib

joblib.dump(scaler, "electricity_scaler.pkl")
```

Load them later:

```python
from tensorflow.keras.models import load_model
import joblib

model = load_model("electricity_lstm.keras")
scaler = joblib.load("electricity_scaler.pkl")
```

This avoids retraining the model whenever a new forecast is needed.

## 📚 What I Learned

### Data Science and Preprocessing

- How to clean real-world time-series data.
- How to handle missing values and incomplete hourly periods.
- How to work with timestamps and Pandas resampling.
- Why chronological splitting is important in time-series forecasting.
- Why preprocessing must avoid data leakage.

### Machine Learning and Deep Learning

- How LSTM networks learn patterns from sequential data.
- How to transform a time series into supervised learning sequences.
- How scaling affects neural network training.
- How to use early stopping to help reduce overfitting.
- How to evaluate regression models using MAE, RMSE, and R².

### Forecasting Concepts

- How a 24-hour lookback window works.
- How to convert scaled predictions back to their original units.
- How recursive multi-step forecasting works.
- Why longer forecasting horizons can increase uncertainty.
- Why a regression model should not be evaluated using classification accuracy alone.

### Practical Engineering

- How to save and reload a trained model.
- How to preserve a fitted scaler for inference.
- How to generate timestamped forecasts.
- How to structure an end-to-end machine learning workflow.

## 🚧 Limitations and Future Improvements

The current model is a baseline forecasting system and has several opportunities for improvement.

- **Add time-based features:** Hour of day, day of week, and month could help the model learn recurring usage patterns.
- **Compare against baselines:** Evaluate persistence forecasting and seasonal-naive forecasting before claiming the LSTM is better.
- **Improve multi-step evaluation:** Measure errors separately for 1-hour, 6-hour, 12-hour, and 24-hour horizons.
- **Improve visualizations:** Add actual-versus-predicted plots, residual plots, and training-versus-validation loss curves.
- **Experiment with architectures:** Compare LSTM against GRU, simpler recurrent models, and tree-based approaches.
- **Build an interface:** Add user-selected forecast horizons and clear tables or charts.
- **Improve forecasting robustness:** Handle gaps and validate the latest 24 timestamps before inference.

## 🚀 Potential Applications

- 🏠 Household energy monitoring
- ⚡ Electricity demand analysis
- 📈 Energy usage planning
- 🏢 Building energy management research
- 🌱 Energy efficiency studies

These are potential applications. The current project forecasts household consumption; it does not yet provide live smart-meter integration or an operational energy-management system.

## 🎤 Project Pitch

**Short pitch:**

"Electricity consumption is difficult to predict because usage patterns change over time. I built an electricity consumption forecasting system using an LSTM neural network that learns from historical household energy readings and predicts future hourly consumption. The project covers the complete machine learning workflow, from cleaning real-world data and creating time-series sequences to training, evaluation, and recursive multi-hour forecasting. My next goal is to improve forecast reliability by incorporating time-based features and comparing the model against simpler forecasting baselines."

### 💡 What Makes This Project Valuable?

It combines data engineering, time-series analysis, deep learning, and model deployment fundamentals in one practical project. It also demonstrates how a trained model can be used repeatedly to generate future predictions rather than only producing results during training.

## 🏁 Conclusion

This project demonstrates an end-to-end deep learning workflow for household electricity consumption forecasting using historical data.

It establishes a foundation for more advanced forecasting systems by combining hourly aggregation, leakage-aware preprocessing, LSTM training, quantitative evaluation, model persistence, and recursive multi-hour prediction.

---

<p align="center">
  <b>⚡ Turning Historical Energy Data into Future Insights ⚡</b>
</p>

<p align="center">
  Built with Python, TensorFlow, and a passion for Machine Learning.
</p>
