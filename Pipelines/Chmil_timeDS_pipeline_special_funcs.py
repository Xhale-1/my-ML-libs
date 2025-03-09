
from os import replace
import numpy as np
import numpy as np

    

def shuffle_within_hundreds(arr):
        # Определяем количество полных сотен
        n_hundreds = len(arr) // 100
        remainder = len(arr) % 100  # Остаток, если длина не кратна 100
        
        # Копируем массив для изменений
        shuffled = arr.copy()
        
        # Перемешиваем каждую сотню
        for i in range(n_hundreds):
            start = i * 100
            end = (i + 1) * 100
            np.random.shuffle(shuffled[start:end])
        
        # Перемешиваем остаток, если он есть
        if remainder > 0:
            np.random.shuffle(shuffled[n_hundreds * 100:])
        
        return shuffled



def special_split(x, shuffle = 0):

    if not isinstance(x,np.ndarray):
      x = np.array(x)

    ids = np.arange(0, x.shape[0] - 1)
    near1 = round(int(0.67 * len(ids)) / 100) * 100
    near2 = round(int(0.83 * len(ids)) / 100) * 100
    tr, vl, ts = np.split(ids, [near1, near2])

    if shuffle:
      tr = shuffle_within_hundreds(tr)
      vl = shuffle_within_hundreds(vl)
      ts = shuffle_within_hundreds(ts)
    
    return tr, vl, ts

def shuffle(ids):
  ids = np.random.choice(ids,len(ids), replace = False)
  return ids





import torch
import matplotlib.pyplot as plt
import math



def special_preds(model, x, y, n_graphs=1):
    
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x,dtype = torch.float32)
    if not isinstance(y, torch.Tensor):
      y = torch.tensor(y,dtype = torch.float32)
    
    ids = np.arange(0,x.shape[0],100)
    ids2 = np.random.choice(ids, n_graphs, replace=False)
    
    x_i = []
    y_i = []
    x_pred_i = []
    preds_i = []
    for i in ids2:

      base = x[i, :4]
      time = torch.linspace(x[i:i+100, 4].min(), x[i:i+100, 4].max(), 1000)
      new_rows = [torch.cat((base, j.unsqueeze(0))) for j in time]  # Создаем новые строки
      x1 = torch.stack(new_rows).reshape(-1, 5)  # Формируем новый тензор

      loader = torch.utils.data.DataLoader(x1, batch_size=int(0.05 * x1.shape[0]), 
                                         shuffle=False, num_workers=2)
      model.eval()
      preds = []
      with torch.no_grad():
        for batch in loader:
            pred = model(batch)
            preds.append(pred)
        preds = torch.cat(preds).detach()

      x_i.append(x[i:i+100,4].reshape(-1,1))
      y_i.append(y[i:i+100].reshape(-1,1)) 
      x_pred_i.append(time.reshape(-1,1))
      preds_i.append(preds.reshape(-1,1))

    return x_i, y_i, x_pred_i, preds_i





def special_plot(x_i, y_i, x_pred_i, preds_i, n_cols=2):

    n_plots = len(preds_i)
    n_rows = math.ceil(n_plots / n_cols) 
    
    # Создаём сетку графиков
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    
    # Если только один график, axs не будет массивом, преобразуем
    if n_plots == 1:
        axs = [[axs]]  # Делаем вложенный список для единообразия
    # Если одна строка или один столбец, корректируем индексацию
    elif n_rows == 1:
        axs = [axs]  # Преобразуем в список для итерации
    elif n_cols == 1:
        axs = [[ax] for ax in axs]  # Преобразуем в список списков

    # Отрисовка графиков
    for i in range(n_plots):
        row = i // n_cols  # Номер строки
        col = i % n_cols   # Номер столбца

        axs[row][col].scatter(x_pred_i[i], preds_i[i], s=2) #, label='Predictions')
        axs[row][col].scatter(x_i, y_i, s=5) #, label='True Data')
        axs[row][col].set_title(f'График {ids2_list[i][0]}')
        axs[row][col].set_xlabel('time, %')
        axs[row][col].set_ylabel('y')
        #axs[row][col].legend()

    # Убираем лишние пустые графики, если их больше, чем n_plots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row][col])

    plt.tight_layout()  # Улучшает расположение
    plt.show()
