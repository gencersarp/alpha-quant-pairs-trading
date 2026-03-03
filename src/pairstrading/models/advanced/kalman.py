import numpy as np


class KalmanHedgeRatio:
    """Estimates time-varying hedge ratio using a 1D Kalman Filter."""
    def __init__(self, trans_cov: float = 1e-5, obs_cov: float = 1e-3):
        self.trans_cov = trans_cov
        self.obs_cov = obs_cov

    def fit(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Fits the Kalman filter to find the dynamic beta where y = beta * x.
        """
        n = len(y)
        beta = np.zeros(n)
        p_cov = np.zeros(n)
        
        # Initial state guess
        beta[0] = y[0] / x[0] if x[0] != 0 else 0
        p_cov[0] = 1.0

        for t in range(1, n):
            # Predict step
            beta_pred = beta[t - 1]
            p_pred = p_cov[t - 1] + self.trans_cov

            # Update step
            y_pred = beta_pred * x[t]
            error = y[t] - y_pred
            
            # Variance of prediction error
            s_var = x[t]**2 * p_pred + self.obs_cov
            
            # Kalman Gain
            k_gain = p_pred * x[t] / s_var if s_var != 0 else 0
            
            beta[t] = beta_pred + k_gain * error
            p_cov[t] = (1 - k_gain * x[t]) * p_pred

        return beta