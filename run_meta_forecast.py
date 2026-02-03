
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from model import LearningCurveForecaster

def simulate_curve(steps, a_range=(1.0, 5.0), b_range=(0.3, 0.5), c_range=(0.01, 0.1)):
    """Generates a synthetic curve with random params"""
    a = np.random.uniform(*a_range)
    b = np.random.uniform(*b_range)
    c = np.random.uniform(*c_range)
    # y = a * x^-b + c
    clean = a * (steps**-b) + c
    # Noise depends on magnitude
    noise = np.random.normal(0, 0.02 * clean)
    return clean + noise

def main():
    np.random.seed(42)
    
    # --- Step 1: Meta-Training ---
    print("1. Generating and processing historical curves...")
    # Use logarithmic spacing for training steps to cover the learning process well
    train_steps = np.unique(np.geomspace(10, 1000, 20).astype(int))
    
    n_history = 50
    hist_thetas = []
    
    print(f"   Fitting GP to {n_history} historical curves...")
    for i in range(n_history):
        losses = simulate_curve(train_steps)
        forecaster = LearningCurveForecaster()
        forecaster.fit(train_steps, losses)
        # Collect the learned log-transformed hyperparameters (theta)
        hist_thetas.append(forecaster.model.kernel_.theta)
        
    # Average the hyperparameters to get a "robust" prior
    avg_theta = np.mean(hist_thetas, axis=0)
    print(f"   Learned average hyperparameters (theta): {avg_theta}")
    
    # Create a kernel instance with these optimized parameters
    # We use a dummy instance to access the kernel structure
    dummy = LearningCurveForecaster()
    best_kernel = dummy.model.kernel.clone_with_theta(avg_theta)
    print(f"   Best Kernel: {best_kernel}")

    # --- Step 2: Forecasting a New Curve ---
    print("\n2. Forecasting a new partial curve...")
    
    # True future for evaluation
    full_steps = np.linspace(10, 3000, 300)
    # Generate a new curve from the same distribution
    new_losses = simulate_curve(full_steps, a_range=(2.0, 3.0), b_range=(0.35, 0.45))
    
    # Observe only a small prefix (e.g., first 6 points) - challenging for standard GP
    obs_n = 6
    obs_steps = full_steps[:obs_n]
    obs_losses = new_losses[:obs_n]
    
    print(f"   Observed {obs_n} points. Predicting up to step {int(full_steps[-1])}...")

    # Method A: Baseline (Default initialization, full optimization)
    # With few points, this might overfit or revert to prior mean poorly
    model_a = LearningCurveForecaster()
    model_a.fit(obs_steps, obs_losses)
    preds_a, std_a = model_a.predict(full_steps)

    # Method B: Meta-Learning (Fixed "best" hyperparameters)
    # We turn off the optimizer to enforce the learned priors strictly
    model_b = LearningCurveForecaster(kernel=best_kernel, optimizer=None)
    model_b.fit(obs_steps, obs_losses)
    preds_b, std_b = model_b.predict(full_steps)

    # --- Step 3: Visualization ---
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    
    # Plot Truth
    plt.plot(full_steps, new_losses, 'k--', label='True Future (Hidden)', alpha=0.6, lw=1.5)
    plt.scatter(obs_steps, obs_losses, c='black', s=60, zorder=5, label='Observed Prefix')
    
    # Plot Baseline
    plt.plot(full_steps, preds_a, color='red', label='Baseline (Default)')
    plt.fill_between(full_steps, preds_a - 1.96*std_a, preds_a + 1.96*std_a, color='red', alpha=0.1)

    # Plot Meta
    plt.plot(full_steps, preds_b, color='forestgreen', lw=2, label='Meta-Learned (Fixed Params)')
    plt.fill_between(full_steps, preds_b - 1.96*std_b, preds_b + 1.96*std_b, color='forestgreen', alpha=0.1)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Meta-Learning GP Forecast (N_obs={obs_n})')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    output_file = 'meta_learning_result.png'
    plt.savefig(output_file)
    print(f"\nSaved comparison plot to {output_file}")

if __name__ == "__main__":
    main()
