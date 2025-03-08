
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from scipy.stats import chi2




class outs:
    
    def outs_with_iso(x00, y00):
      iso = IsolationForest( contamination = 0.03, random_state=42)
      outliers = iso.fit_predict(x00)
    
      mask = outliers == 1
      #print(len(x00[mask]))
      return x00[mask], y00[mask] 
    
    
    def outs_with_advanced_mahalanobis(X, torch_out = 0):
    
      if not isinstance(X, np.ndarray):
        X = np.array(X)
      
      # 1. Применяем MCD для получения робастных оценок
      mcd = MinCovDet(random_state=42, support_fraction=0.7)  # support_fraction — доля "чистых" данных (0.5–1)
      mcd.fit(X)
    
      # Робастное среднее и ковариационная матрица
      robust_mean = mcd.location_  # \( \mu_{MCD} \)
      robust_cov = mcd.covariance_  # \( \Sigma_{MCD} \)
    
      # Инвертируем ковариационную матрицу
      robust_cov_inv = np.linalg.inv(robust_cov)
    
      # 2. Вычисляем расстояние Махаланобиса для каждой точки
      mahal_distances = np.array([distance.mahalanobis(x, robust_mean, robust_cov_inv)**2 for x in X])
    
      # 3. Устанавливаем порог (95% для 4 признаков)
      threshold = chi2.ppf(0.95, df=4)  # 4 степени свободы (количество признаков)
    
      # 4. Находим выбросы
      outlier_indices = np.where(mahal_distances > threshold)[0]
    
      # Опционально: извлекаем выбросы для анализа
      outliers = X[outlier_indices]
    
      all_ids = range(len(X))
      non_outs = list(set(all_ids) - set(outlier_indices))
      X_non_outs = X[non_outs]
    
      if torch_out:
        X_non_outs = torch.tensor(X_non_outs, dtype = torch.float32)
    
      return X_non_outs, outlier_indices
