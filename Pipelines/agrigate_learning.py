
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np
import matplotlib.pyplot as plt



# __________________HYPEROPT_____________________

# Определяем целевую функцию для Hyperopt
def objective(params):


    # Запускаем pipe1 с параметрами
    rmse = pipe1(
        data=data,  # Ваш DataFrame
        net=net,
        feat=5,
        bs=params["bs"],
        lr0=params["lr0"],
        lr1=params["lr1"],
        eps=7,
        device=device
    )

    return {
        "loss": rmse,
        "status": STATUS_OK
    }

# Пространство гиперпараметров
space = {
    "bs": hp.choice("bs", [0.02, 0.08, 0.1, 0.3]),
    "lr0": hp.choice("lr0", [0.01, 0.1, 1]),
    "lr1": hp.choice("lr1", [0.0001, 0.001, 0.01, 0.1]),
}

# Запуск оптимизации
trials = Trials()  # Для хранения истории испытаний
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,  # Байесовская оптимизация с TPE
    max_evals=20,      # Количество испытаний (можно увеличить)
    trials=trials
)

# Преобразование индексов в значения
param_values = {
    "bs": [0.02, 0.08, 0.1, 0.3],
    "lr0": [0.01, 0.1, 1],
    "lr1": [0.0001, 0.001, 0.01, 0.1],
}

best_params = {key: param_values[key][best[key]] for key in best}
print("\nЛучшие параметры:")
print(best_params)

# Лучший результат
best_val_mse = min([trial["result"]["loss"] for trial in trials.trials])
print(f"Best Validation MSE: {best_val_mse:.4f}")

# Визуализация истории оптимизации
losses = [trial["result"]["loss"] for trial in trials.trials]
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, marker='o')
plt.title("Validation MSE по испытаниям")
plt.xlabel("Испытание")
plt.ylabel("Validation MSE")
plt.grid(True)
plt.show()









#_______________OPTUNA___________________

!pip install optuna

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_objective(print_loss, feat, eps):
  def objective(trial):
      n_layers = trial.suggest_int("n_layers", 1, 15)
      layer_configs = [trial.suggest_int(f"layer_{i+1}_neurons", 1, 250) for i in range(n_layers)]

      #batch_size = trial.suggest_float("batch_size", 0.005, 0.3)
      batch_size = 0.015
      #lr0 = trial.suggest_float("lr0", 0.01, 1.0)
      lr0 = 0.018
      #lr1 = trial.suggest_float("lr1", 0.00001, 0.01)
      lr1 = 0.0003

      model = DynamicNet(4,layer_configs,1, device)
      loss = pipe2(model, data, feat=feat, bs=batch_size, lr0=lr0, lr1=lr1, eps=eps, device=device, print_loss = print_loss)

      return loss
  return objective




for feat in range(8,9,1):
  study = optuna.create_study(directions=["minimize"])
  study.optimize(make_objective(print_loss=0, feat = feat, eps = 70), n_trials=100)

  print("_________________________________")
  print(f"feature: {feat}")
  successful_trials = [t for t in study.trials if t.values]
  best_trials = sorted(successful_trials, key=lambda t: t.values[0])[:5]
  for i, trial in enumerate(best_trials, 1):
      print(f"Модель {i}: Слои={trial.params['n_layers']}, Loss={trial.values[0]:.6f}")
      print("Конфигурация слоев:", [trial.params[f"layer_{j+1}_neurons"] for j in range(trial.params['n_layers'])])
  print("__________________________________")
