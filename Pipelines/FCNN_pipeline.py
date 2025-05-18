
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import sklearn
from sklearn.metrics import root_mean_squared_error
import shutil
from sklearn.linear_model import LinearRegression
from tqdm import tqdm




#____________IMPORT_____________________
#_______________________________________


def split_train(x00, proportions):
    """
    Разделяет массив индексов на части согласно заданным пропорциям.

    Параметры:
        x00: исходный массив данных (или его длина).
        proportions: список пропорций для разбиения (например, [0.6, 0.8, 0.9]).

    Возвращает:
        Кортеж массивов индексов.
    """
    ids = np.random.choice(len(x00), len(x00), replace=False) # Генерация случайных индексов без повторений
    split_indices = [int(prop * len(x00)) for prop in proportions] #создание листа индексов пропорций
    parts = np.split(ids, split_indices)
    
    return parts






def scale(objs, scale_data=None):
    """
    Масштабирует данные с использованием StandardScaler.

    :param objs: Список объектов (массивов или списков), которые нужно масштабировать.
    :param scale_data: Список объектов (массивов или списков), на которых обучается scaler.
                       Если None, используется `objs` для обучения scaler.
    :return: Кортеж из двух элементов:
             - scaled_objs: Список масштабированных объектов (тензоры PyTorch).
             - scalers: Список обученных StandardScaler.
    """
    scalers = []
    scaled_objs = []

    # Если scale_data не предоставлено, используем objs для обучения scaler
    if scale_data is None:
        scale_data = objs


    for x00, scale_data_x in zip(objs, scale_data):

        if not isinstance(x00, np.ndarray):
            x00 = np.array(x00)
            if len(x00.shape) == 1:
               x00 = x00.reshape(-1,1)
        if not isinstance(scale_data_x, np.ndarray):
            scale_data_x = np.array(scale_data_x)
            if len(scale_data_x.shape) == 1:
               scale_data_x = scale_data_x.reshape(-1,1)

        scaler = StandardScaler()
        scaler.fit(scale_data_x)

        x_scaled = scaler.transform(x00)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        scalers.append(scaler)
        scaled_objs.append(x_tensor)

    return scaled_objs, scalers





def descale(x, scaler, col=-1):
    """
    Дескейлит данные, используя StandardScaler.
    :param x: Данные для дескейла (массив или тензор).
    :param scaler: Обученный StandardScaler.
    :param col: Индекс колонки для дескейла. Если -1, дескейлит весь массив.
    :return: Дескейленные данные.
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    
    if col == -1:
        # Дескейлим весь массив
        x1 = scaler.inverse_transform(x)
    else:
        # Дескейлим только указанную колонку
        mean = scaler.mean_[col]  
        std = scaler.scale_[col]  
        x1 = x * std + mean  
    
    return x1



def loaders(x,y,prop_array, bs = 0.01, addtr = 0):

  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x, dtype = torch.float32)
  if not isinstance(y, torch.Tensor):
    y = torch.tensor(y, dtype = torch.float32)
  

  loaders = []

  for i,prop in enumerate(prop_array):
    x_prop = x[prop]
    y_prop = y[prop]

    fakeds = torch.utils.data.TensorDataset(x_prop, y_prop)

    shuffle = (i == 0)
    batch_size = int(bs * len(fakeds)) if i == 0 else int(2.5 * bs * len(fakeds))
    
    loader = torch.utils.data.DataLoader(
        fakeds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    loaders.append(loader)

    if addtr and i==0:
        loader = torch.utils.data.DataLoader(
        fakeds,
        batch_size=int(2.5 * bs * len(fakeds)),
        shuffle=False,
        num_workers=2
        )
        loaders.append(loader)

  return loaders





#_________________MODEL____________________
#__________________________________________


class DynamicNet(nn.Module):
  def __init__(self, inputshape, array, outputshape, device):
    super().__init__()

    self.layers = nn.Sequential()
    self.act = nn.LeakyReLU()

    prev_num = inputshape
    for i in range(len(array)):
      self.layers.append(nn.Linear(prev_num, array[i]))
      self.layers.append(self.act)
      prev_num = array[i]
    

    self.layers.append(nn.Linear(array[-1], outputshape))

    self.to(device)
      

    
  def forward(self, x):
    x = self.layers(x)
    return x

#model = net([10,12])
#print(model(torch.randn(10,12)))







#____________________LEARNING________________________
#____________________________________________________



def learning(trloader, 
             vlloader, 
             criterion, 
             model, 
             optimizer, 
             eps, 
             device = 'cpu', 
             sch = None, 
             print_loss = 1, 
             earlystop = 0, 
             extr_slope = -0.005):

    loaders = {"train": trloader, "valid": vlloader}
    avg_losses = {"train": [], "valid": []}
    if earlystop:
      err_num = 0

    for epoch in tqdm(range(eps)):
        for k, loader in loaders.items():
            total_loss = 0  # Для подсчёта среднего MSE


            for batch in loader:
                if k == 'train':
                    model.train()
                    pred = model(batch[0].to(device))
                    loss = criterion(pred, batch[1].to(device))

                    total_loss += loss.cpu().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if k == 'valid':
                    model.eval()
                    pred = model(batch[0].to(device))
                    loss = criterion(pred, batch[1].to(device))
                    total_loss += loss.cpu().item()

            average_loss = total_loss / len(loaders[k])
            avg_losses[k].append(average_loss)

            if(print_loss):
              print(f"pred: {pred[0].cpu().detach().numpy().round(3)}, loss для {k}: {average_loss}")
              print(f"true: {batch[1][0].cpu().numpy().round(3)}")

        if not sch is None:
          sch.step()
        
        if earlystop:
          err_num = err_num + 1
          if  err_num == earlystop:
            early = avg_losses['valid'][-earlystop:]
            x_regr = np.arange(len(early)).reshape(-1, 1)
            y_regr = np.array(early)
            y_regr_norm = (y_regr - np.mean(y_regr)) / (np.std(y_regr) + 1e-9)  
            reg = LinearRegression().fit(x_regr, y_regr_norm)
            slope = reg.coef_[0]
            err_num = 0
            if slope > extr_slope:
              current_lr = optimizer.param_groups[0]['lr']
              print(f'early stopped at ep:{epoch}, lr={current_lr}, vl_loss={avg_losses["valid"][-1]}, slope = {slope}')
              break
    
    return avg_losses



def err_plot(losses):
  plt.plot(losses['train'], c = 'red')
  plt.plot(losses['valid'])
  plt.yscale('log')
  plt.legend()
  plt.show()



def save_to_drive(model,name, path):
  torch.save(model.state_dict(), f"/content/{name}.pth")
  shutil.copy(f"/content/{name}.pth", path)






#_________INFERENCE________________________
#_____________________________________________


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



def inference(model, loaders, ys, device, scaler2 = 0, print_loss = 1):
  
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

    rmse0 = root_mean_squared_error(y1_tr, preds1_tr)
    if print_loss:
      print(f'rmse test: {rmse0}')
    
    relerr = ( np.abs(y1_tr -  preds1_tr) / y1_tr ) *100
    maxrelerr = relerr.max()
    print(f'макс относительная ошибка (%) = {maxrelerr} ({i})')
    
  return preds_tr, preds1_tr, rmse0
