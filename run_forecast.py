# /// script
# dependencies = [
#   "numpy",
#   "scikit-learn",
#   "matplotlib",
# ]
# ///

import numpy as np
import matplotlib.pyplot as plt
from model import LearningCurveForecaster

def simulate_learning_curve(steps):
    """Generates a synthetic power-law curve: y = a * x^-b + c"""
    a, b, c = 2.5, 0.35, 0.05
    clean_loss = a * (steps**-b) + c
    # Add heteroscedastic noise (noise decreases as loss decreases)
    noise = np.random.normal(0, 0.02 * clean_loss)
    return clean_loss + noise

def main():
    # 1. Setup Data
    train_steps = np.array([10, 20, 40, 80, 160, 320, 640])
    train_losses = simulate_learning_curve(train_steps)
    
    # Future horizon for forecasting
    future_steps = np.linspace(10, 5000, 200)

    # 2. Fit and Predict
    forecaster = LearningCurveForecaster()
    print(f"Fitting model on {len(train_steps)} points...")
    forecaster.fit(train_steps, train_losses)
    
    preds, stds = forecaster.predict(future_steps)

    # 3. Visualization
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.scatter(train_steps, train_losses, color='black', zorder=5, label='Observed Loss')
    
    # Plot forecast
    plt.plot(future_steps, preds, color='#1f77b4', lw=2, label='GP Forecast (Median)')
    
    # Plot uncertainty bounds (95% confidence interval)
    plt.fill_between(
        future_steps, 
        np.maximum(0, preds - 1.96 * stds), 
        preds + 1.96 * stds, 
        color='#1f77b4', 
        alpha=0.2, 
        label='95% Confidence'
    )

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Learning Curve Forecasting (Log-Log Scale)')
    plt.xlabel('Steps (Log)')
    plt.ylabel('Loss (Log)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    print("Forecast complete. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()