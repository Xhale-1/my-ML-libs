
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import MinCovDet
from scipy.spatial import distance
from scipy.stats import chi2




def analyze_residuals(y_ts, preds_ts):
      """
      Analyze residuals by creating scatter plot, histogram, and calculating statistics.
      
      Parameters:
      y_ts : tensor or array - True values
      preds_ts : tensor or array - Predicted values

      """

      if not isinstance(y_ts, np.ndarray):
        y_ts = np.array(y_ts)

      if not isinstance(preds_ts, np.ndarray):
        if isinstance(preds_ts, torch.Tensor):
          preds_ts = preds_ts.detach()
        preds_ts = np.array(preds_ts)

      # Calculate residuals
      residuals = y_ts.flatten() - preds_ts.flatten()
      
      # Detach residuals if it's a tensor
      residuals_detached = residuals.detach() if torch.is_tensor(residuals) else residuals
      
      # Residuals vs Predicted values scatter plot
      plt.figure(figsize=(5, 3))
      plt.scatter(preds_ts.flatten(), residuals_detached)
      plt.xlabel('Predicted values')
      plt.ylabel('Residuals')
      plt.axhline(y=0, color='r', linestyle='--')
      plt.title(f'Residuals vs Predicted values')
      plt.show()
      
      # Histogram of residuals
      plt.figure(figsize=(5, 3))
      plt.hist(residuals_detached, bins=50)
      plt.xlabel('Residuals')
      plt.ylabel('Frequency')
      plt.title(f'Histogram of Residuals')
      plt.show()
      
      # Calculate statistics
      mean_residuals = residuals_detached.mean()
      std_residuals = torch.std(residuals) if torch.is_tensor(residuals) else np.std(residuals)
      
      print(f"Mean of residuals: {mean_residuals}")
      print(f"Standard deviation of residuals: {std_residuals}")

      # Example usage:
      # analyze_residuals(y[ts], preds_ts)


      
