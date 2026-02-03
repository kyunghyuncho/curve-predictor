# Learning Curve Forecaster

A modular Python repository for high-fidelity forecasting of machine learning training progress.

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
Ensure you have `uv` installed. You can run the demo directly:

```bash
uv run run_forecast.py
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