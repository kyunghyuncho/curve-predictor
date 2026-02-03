import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from typing import Tuple, Optional

class LearningCurveForecaster:
    """
    A non-parametric forecaster for learning curves using Gaussian Processes.
    Fits in log-log space to capture power-law dynamics (y = ax^-b).
    """
    def __init__(self, nu: float = 1.5, kernel=None, optimizer="fmin_l_bfgs_b"):
        # Matern kernel (nu=1.5) is less smooth than RBF, 
        # making it better for noisy SGD dynamics.
        if kernel is None:
            self.kernel = (
                C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=nu) + 
                WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
            )
        else:
            self.kernel = kernel

        self.model = GaussianProcessRegressor(
            kernel=self.kernel, 
            n_restarts_optimizer=10,
            alpha=0.0,
            optimizer=optimizer
        )
        self.is_fitted = False

    def _to_log_space(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms data to log-log space for linearization."""
        return np.log(x).reshape(-1, 1), np.log(y)

    def fit(self, steps: np.ndarray, losses: np.ndarray):
        """Fits the GP model to the provided training history."""
        if len(steps) < 3:
            raise ValueError("At least 3 observations are required for forecasting.")
        
        x_log, y_log = self._to_log_space(steps, losses)
        self.model.fit(x_log, y_log)
        self.is_fitted = True
        return self

    def predict(self, future_steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts future loss values.
        Returns:
            mu: Predicted loss (median of the log-normal distribution)
            std: Estimated standard deviation in the original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        x_future_log = np.log(future_steps).reshape(-1, 1)
        y_pred_log, y_std_log = self.model.predict(x_future_log, return_std=True)

        # Map back to original space
        # Note: We use the median of the Log-Normal for the point prediction
        y_pred = np.exp(y_pred_log)
        
        # Law of total variance for Log-Normal distribution
        y_var = (np.exp(y_std_log**2) - 1) * np.exp(2 * y_pred_log + y_std_log**2)
        y_std = np.sqrt(y_var)

        return y_pred, y_std