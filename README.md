# Learning Curve Forecaster

A modular Python repository for high-fidelity forecasting of machine learning training progress.

It features a **Meta-Learning** approach that leverages historical training curves to build robust priors, enabling accurate forecasts even from very short partial prefixes (e.g., the first few epochs).

## 1. The Core Strategy: Log-Log Manifold
Most learning curves (loss/error) follow a **Power Law** distribution:
$$y = ax^{-b} + c$$

Standard Gaussian Processes (GPs) struggle with the sharp curvature and strictly positive nature of loss. This library solves this by projecting the data into a **log-log manifold**:
* Linearizes the power-law trend.
* Enforces positivity (predictions in log-space are always positive in original space).
* Handles heteroscedasticity (uncertainty naturally scales with the magnitude of the loss).

## 2. Kernel Selection
We utilize a composite kernel:
* **Matern ($\nu=1.5$):** Captures the non-infinite smoothness of SGD dynamics better than a standard RBF kernel.
* **WhiteNoise:** Explicitly models evaluation jitter, preventing the GP from over-fitting to noisy validation spikes.

## 3. Back-Transformation (The Math)
When predicting, the GP provides a mean $\mu_{log}$ and variance $\sigma^2_{log}$. To return to the original scale, we treat the result as a **Log-Normal Distribution**.
* **Point Prediction:** We use the median, which is $\exp(\mu_{log})$.
* **Uncertainty:** The standard deviation in the original scale is calculated using:
    $$\text{std} = \sqrt{(\exp(\sigma^2_{log}) - 1) \cdot \exp(2\mu_{log} + \sigma^2_{log})}$$

## 4. Usage

### Installation
Install the necessary dependencies:
```bash
uv venv
source ./.venv/bin/activate
uv pip install -r requirements.txt
```

### Basic Forecasting
To run a simple forecast on a single curve:
```bash
python run_forecast.py
```

```python
from model import LearningCurveForecaster
import numpy as np

# Your training data
steps = np.array([100, 200, 400])
losses = np.array([0.5, 0.4, 0.35])

# Forecast
model = LearningCurveForecaster().fit(steps, losses)
horizon = np.array([1000, 2000])
preds, stds = model.predict(horizon)
```

## 5. Meta-Learning for Deep Forecasting
If you need to forecast from very short prefixes (e.g., just the first few epochs), standard fitting is unstable. We provide a **Meta-Learning** approach that learns robust priors from your historical experiments.

### Jupyter Notebook Tool (Recommended)
We provide an interactive notebook to guide you through the process:
```bash
jupyter notebook forecasting_tool.ipynb
```
This notebook allows you to:
1.  Load historical learning curves.
2.  Visualize the distribution of learned kernel parameters.
3.  Compare "Baseline" vs. "Meta-Learned" forecasts on new data.

### CLI Script
You can also run the meta-learning demo directly:
```bash
python run_meta_forecast.py
```
This script acts as a proof-of-concept, simulating historical data and generating a comparison plot `meta_learning_result.png`.