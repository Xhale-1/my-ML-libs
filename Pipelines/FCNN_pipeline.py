
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import sklearn
import shutil
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from copy import deepcopy




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
  
  if prop_array == None:
    prop_array = [list(range(x.shape[0]))]
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
             earlystop = 0, patience = 14,
             autopilot = 0,
             overfit = 0, of_break = 1, lr_surge = 3,
             bestmodel = 0):

    loaders = {"train": trloader, "valid": vlloader}
    avg_losses = {"train": [], "valid": []}
    if autopilot:
      au_window = 0
      au_slope = 0
      batch_loss = []
    if overfit:
      of_window = 0
    if earlystop or bestmodel: 
        best_valid_loss = np.inf
        no_improve = 0
        best_model_state = None 

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
                    batch_loss.append(loss.cpu().item())

            average_loss = total_loss / len(loaders[k])
            avg_losses[k].append(average_loss)

            if(print_loss):
              print(f"pred: {pred[0].cpu().detach().numpy().round(3)}, loss для {k}: {average_loss}")
              print(f"true: {batch[1][0].cpu().numpy().round(3)}")


        if earlystop or bestmodel:  # Объединённая проверка
            current_valid_loss = avg_losses['valid'][-1]
            
            if current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                no_improve = 0
                
                if bestmodel:  # Сохраняем модель только если bestmodel=True
                    best_model_state = deepcopy(model.state_dict())
                    if print_loss:
                      print(f"New best model! Val loss: {best_valid_loss:.4f}")
            else:
                no_improve += 1
            
            # Ранняя остановка (если earlystop=True)
            if earlystop and no_improve >= patience:
                print(f'\nEarly stopped at ep:{epoch}, because {patience} eps no better vl_los')
                break
            


        if autopilot:
          au_window += 1
          if  au_window == autopilot:
            au_window = 0

            lastlossdata = batch_loss
            batch_loss = []
            #lastlossdata = avg_losses['valid'][-autopilot:]
            x_regr = np.arange(len(lastlossdata)).reshape(-1, 1)
            y_regr = np.array(lastlossdata)
            y_regr_norm = (y_regr - np.mean(y_regr)) / (np.std(y_regr) + 1e-9)  
            reg = LinearRegression().fit(x_regr, y_regr_norm)
            au_slope = reg.coef_[0]

            if print_loss:
              print(' ')
              print(f'au_slope = {au_slope}')
              current_lr = optimizer.param_groups[0]['lr']
              print(f'lr = {current_lr}')


        if not sch is None:
          if autopilot:
            #if au_slope > au_extr_slope and au_slope <= 0:
            if au_slope > 0:
              au_slope = 0
              sch.step()
              if print_loss:
                print(f'sch step and lr ={sch.get_last_lr()}')
          else:
            sch.step()
        

        if overfit:
            if not earlystop:
                patience = 0
                best_valid_loss = np.inf
                
            # state = {
            #     'of_window': of_window, 
            #     'patience': patience, 
            #     'best_valid_loss': best_valid_loss
            # }
            
            # lr_was_adjusted = check_overfit_and_adjust_lr(
            #     optimizer=optimizer, 
            #     avg_losses=avg_losses, 
            #     overfit=overfit, 
            #     lr_surge=lr_surge,
            #     of_break = of_break,
            #     state=state
            # )

            # # Обновляем глобальные переменные из state
            # of_window = state['of_window']
            # patience = state['patience']
            # best_valid_loss = state['best_valid_loss']

            of_window += 1
            if of_window == overfit:
              of_window = 0
              last_valid_losses = np.array(avg_losses['valid'][-overfit:])
              last_train_losses = np.array(avg_losses['train'][-overfit:])
              all_valid_higher = np.all(last_valid_losses > last_train_losses)
              if all_valid_higher:
                  if of_break:
                    print(f'early stopped at ep:{epoch} because {overfit} eps vl_loss > tr_loss')
                    break

    if bestmodel and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model weights")

    return avg_losses, model


# def check_overfit_and_adjust_lr(optimizer, 
#                                 avg_losses, 
#                                 overfit, 
#                                 lr_surge,
#                                 of_break,
#                                 state):
#     """
#     Проверяет, переобучается ли модель, и увеличивает LR, если valid_loss > train_loss на протяжении `overfit` эпох.
    
#     Параметры:
#     - optimizer: оптимизатор (PyTorch)
#     - avg_losses: словарь с 'train' и 'valid' лоссами (например, {'train': [0.5, 0.4, ...], 'valid': [0.6, 0.5, ...]})
#     - overfit: количество эпох для проверки переобучения
#     - lr_surge: множитель для увеличения LR (например, 1.5)
    
#     Возвращает:
#     - lr_was_adjusted: bool (True, если LR был изменён)
#     """
#     state['of_window'] += 1
#     lr_was_adjusted = False
    
#     if state['of_window'] == overfit:
#         state['of_window'] = 0
#         last_valid_losses = np.array(avg_losses['valid'][-overfit:])
#         last_train_losses = np.array(avg_losses['train'][-overfit:])
        
#         # Проверяем, что ВСЕ valid_loss > train_loss
#         all_valid_higher = np.all(last_valid_losses > last_train_losses)
        
#         if all_valid_higher:
#             if of_break:
#               return
#             if state['patience']:
#               state['patience'] = 0
#             if state['best_valid_loss']:
#               state['best_valid_loss'] = np.inf

#             cur_lr = optimizer.param_groups[0]['lr']
#             new_lr = lr_surge * cur_lr
            
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = new_lr
            
#             print(f"[Overfit detected] LR увеличен в {lr_surge}x: {cur_lr:.2e} → {new_lr:.2e}")
#             lr_was_adjusted = True
    
#     return lr_was_adjusted


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




def inference(model, loaders, ys, device, metrics_func, scaler2 = 0, print_loss = 1):
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

    rmse0 = np.sqrt(((y1_tr -  preds1_tr)**2).mean())
    if print_loss:
      print(f'rmse test: {rmse0}')
    

    result = metrics_func(y1_tr, preds1_tr,i)
    metrics.append(result)

  return preds_tr, preds1_tr, rmse0, metrics
