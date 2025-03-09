
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









import torch
import matplotlib.pyplot as plt
import math



def preds_uniq(model, x, y, n_plots=1, n_cols=2):
    # Оптимизируем создание x1
    base = x[0, :4]  # Базовые значения вне цикла
    x1 = torch.tensor([])
    time = torch.linspace(x[:, 4].min(), x[:, 4].max(), 1000)
    for i in time:
        new = torch.cat((base, i.unsqueeze(0)))
        x1 = torch.cat((x1, new), 0)
    x1 = x1.reshape(-1, 5)

    # Загрузка данных и предсказания
    loader = torch.utils.data.DataLoader(x1, batch_size=int(0.05 * x1.shape[0]), 
                                         shuffle=False, num_workers=2)
    model.eval()
    preds = []
    for batch in loader:
        pred = model(batch)
        preds.append(pred.flatten())
    preds = torch.cat(preds)

    # Вычисляем количество строк
    n_rows = math.ceil(n_plots / n_cols)  # Округляем вверх
    
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
        start_idx = i * 100
        end_idx = (i + 1) * 100
        if end_idx > x.shape[0]:  # Проверка выхода за пределы
            break
        axs[row][col].scatter(x1[:, 4].flatten(), preds.flatten().detach(), s=2) #, label='Predictions')
        axs[row][col].scatter(x[start_idx:end_idx, 4], y[start_idx:end_idx], s=5) #, label='True Data')
        #axs[row][col].set_title(f'График {i + 1}')
        axs[row][col].set_xlabel('x[:, 4]')
        axs[row][col].set_ylabel('y')
        #axs[row][col].legend()

    # Убираем лишние пустые графики, если их больше, чем n_plots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axs[row][col])

    plt.tight_layout()  # Улучшает расположение
    plt.show()
