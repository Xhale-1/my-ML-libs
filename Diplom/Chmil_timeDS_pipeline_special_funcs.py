from sklearn.preprocessing import StandardScaler
from os import replace
import numpy as np
import torch


#____________intergral___________
#________________________________



def predict(model,loaders, device, yss=0):
  model.eval()
  preds = []
  ys = []
  out = []
  out_y = []
  for loader in loaders:
    with torch.no_grad():
      for batch in loader:
        pred = model(batch[0].to(device))
        preds.append(pred.cpu())
        if yss:
          ys.append(batch[1])
    preds = torch.cat((preds),0)
    out.append(preds)
    if yss:
      ys = torch.cat((ys),0)
      out_y.append(yss)

  return out, out_y



def inference2(model, loaders, ys, device, scaler2 = 0, print_loss = 1):
  metrics = []
  data = zip(loaders,ys)
  for i,(loader,y) in enumerate(data):
    [preds_tr], _ = predict(model,[loader], device)

    if isinstance(y, torch.Tensor):
      y = y.detach().numpy()
    else:
      y = np.array(y)

    preds1_tr = preds_tr.detach().numpy()
    y1_tr = y
    if isinstance(scaler2, StandardScaler):
      preds1_tr = scaler2.inverse_transform(preds1_tr)
      y1_tr = scaler2.inverse_transform(y)

    if print_loss:
      print(np.array(list(zip(preds1_tr[:5],y1_tr[:5]))).reshape(-1,2))

    mtrcs = []
    rmse0 = np.sqrt(((y1_tr - preds1_tr)**2).mean())
    mtrcs.append(rmse0)
    if print_loss:
      print(f'rmse test: {rmse0}')
    
    relerr = ( np.abs(y1_tr -  preds1_tr) / y1_tr ) *100
    maxrelerr = relerr.max()
    mtrcs.append(maxrelerr)
    print(f'макс относительная ошибка (%) = {maxrelerr} ({i})')


    
    # ids = np.arange(0,y1_tr.shape[0],101)
    # y1_tr_parts = np.split(y1_tr,ids)
    # preds1_tr_parts = np.split(preds1_tr,ids)
    # zip_parts = zip(preds1_tr_parts, y1_tr_parts)
    # rmspe_ls = []
    # maape_ls = []
    # for pr, y1 in zip_parts:
    #   pr = np.array(pr)
    #   y1 = np.array(y1)
    #   rmspe = np.sqrt((((y1 -  pr) / y1)**2).mean())
    #   rmspe_ls.apped(rmspe)
    #   maape = (np.abs((y1 -  pr) / y1)).mean()
    #   maape_ls.append(maape)

    # rmspe_max = np.array(rmspe_ls).max()
    # maape_max = np.array(maape_ls).max()
    # mtrcs.append(rmspe_max)
    # print(f'rmspe: {rmspe_max}')
    # mtrcs.append(maape_max)
    # print(f'maape: {maape_max}'

    chunk_size = 101
    n_chunks = len(y1_tr) // chunk_size + (1 if len(y1_tr) % chunk_size != 0 else 0)
    y1_tr_parts = np.array_split(y1_tr, n_chunks)
    preds1_tr_parts = np.array_split(preds1_tr, n_chunks)
    
    rmspe_ls = []
    maape_ls = []
    
    for pr, y1 in zip(preds1_tr_parts, y1_tr_parts):
        # Убедимся, что массивы не пустые
        if len(y1) == 0 or len(pr) == 0:
            continue
            
        # Защита от деления на ноль
        mask = y1 != 0
        if not np.any(mask):
            continue
            
        y1_part = y1[mask]
        pr_part = pr[mask]
        
        # Вычисляем метрики только для ненулевых значений
        relative_errors = (y1_part - pr_part) / y1_part
        rmspe = np.sqrt(np.mean(relative_errors**2))
        maape = np.mean(np.abs(np.arctan(relative_errors)))  # Для MAAPE обычно используют арктангенс
        
        rmspe_ls.append(rmspe)
        maape_ls.append(maape)
    
    if rmspe_ls and maape_ls:
        rmspe_max = np.max(rmspe_ls)
        maape_max = np.max(maape_ls)
        mtrcs.append(rmspe_max)
        print(f'rmspe: {rmspe_max}')
        mtrcs.append(maape_max)
        print(f'maape: {maape_max}')
    else:
        print("Не удалось вычислить метрики - возможно, все значения y1_tr равны нулю")
    
    metrics.append(mtrcs)
    
  return preds_tr, preds1_tr, metrics










#____________time_________________
#_________________________________

def shuffle_within_hundreds(arr):
        # Определяем количество полных сотен
        n_hundreds = len(arr) // 101
        remainder = len(arr) % 101  # Остаток, если длина не кратна 101
        
        # Копируем массив для изменений
        shuffled = arr.copy()
        
        # Перемешиваем каждую сотню
        for i in range(n_hundreds):
            start = i * 101
            end = (i + 1) * 101
            np.random.shuffle(shuffled[start:end])
        
        # Перемешиваем остаток, если он есть
        if remainder > 0:
            np.random.shuffle(shuffled[n_hundreds * 101:])
        
        return shuffled



def special_split(x, shuffle = 0):

    if not isinstance(x,np.ndarray):
      x = np.array(x)

    ids = np.arange(0, x.shape[0] - 1)
    near1 = round(int(0.67 * len(ids)) / 101) * 101
    near2 = round(int(0.83 * len(ids)) / 101) * 101
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


def special_preds(model, x, y, device, n_graphs=1, scalers=None):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    x_orig = x.numpy()
    y_orig = y.numpy()

    if scalers is not None:
        x_orig = scalers[0].inverse_transform(x_orig)
        y_orig = scalers[1].inverse_transform(y_orig)

    ids = np.arange(0, x.shape[0], 101)
    ids2 = np.random.choice(ids, n_graphs, replace=False)

    x_i, y_i, x_pred_i, preds_i, ids21 = [], [], [], [], []

    model.eval()

    for i in ids2:
        base = x[i, :4]
        time = torch.linspace(x[i:i+101, 4].min(), x[i:i+101, 4].max(), 1000)
        x1 = torch.cat([base.repeat(time.shape[0], 1), time.unsqueeze(1)], dim=1)

        loader = torch.utils.data.DataLoader(x1, batch_size=max(1, int(0.05 * x1.shape[0])), shuffle=False)

        with torch.no_grad():
            preds = torch.cat([model(batch.to(device)).cpu() for batch in loader])

        preds_np = preds.numpy().reshape(-1, 1)

        # Inverse transform predictions
        if scalers is not None:
            preds_np = scalers[1].inverse_transform(preds_np)

            # Inverse transform time component
            time_np = time.unsqueeze(1).numpy()
            # Создаём фиктивные строки с time в 5-й колонке
            time_extended = np.zeros((time_np.shape[0], x.shape[1]))
            time_extended[:, 4] = time_np[:, 0]
            time_np = scalers[0].inverse_transform(time_extended)[:, 4]
            time_np = time_np.reshape(-1, 1)
        else:
            time_np = time.unsqueeze(1).numpy()

        x_i.append(x_orig[i:i+101, 4].reshape(-1, 1))
        y_i.append(y_orig[i:i+101].reshape(-1, 1))
        x_pred_i.append(time_np)
        preds_i.append(preds_np)
        ids21.append(i)

    return x_i, y_i, x_pred_i, preds_i, ids21




def special_plot(x_i, y_i, x_pred_i, preds_i, ids21, n_cols=2):

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
        axs[row][col].scatter(x_i[i], y_i[i], s=5) #, label='True Data')
        axs[row][col].set_title(f'График {ids21[i]}')
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
