
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
import sklearn
import shutil




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





def scale(x00,y00, scale_data = 0, scale_data_x = 0, scale_data_y = 0 ):

  if not isinstance(x00, np.ndarray):
    x00 = np.array(x00)
  if not isinstance(y00, np.ndarray):
    y00 = np.array(y00)
  if not isinstance(scale_data_x, np.ndarray):
    scale_data_x = np.array(scale_data_x)
  if not isinstance(scale_data_y, np.ndarray):
    scale_data_y = np.array(scale_data_y)

  scaler1 = StandardScaler()
  scaler2 = StandardScaler()

  DATAX = x00
  DATAY = y00
  if scale_data:
    DATAX = scale_data_x
    DATAY = scale_data_y

  scaler1.fit(DATAX)
  scaler2.fit(DATAY)
  x0 = scaler1.transform(x00)
  y0 = scaler2.transform(y00)

  x = torch.tensor(x0, dtype = torch.float32)
  y = torch.tensor(y0, dtype = torch.float32).reshape(-1,1)

  return x,y,scaler1,scaler2





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



def loaders(x,y,prop_array, bs = 0.01):

  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x, dtype = torch.float32)
  if not isinstance(y, torch.Tensor):
    y = torch.tensor(y, dtype = torch.float32)
  

  fakedss = []
  loaders = []

  for prop in prop_array:
    x_prop = x[prop]
    y_prop = y[prop]

    fakeds = torch.utils.data.TensorDataset(x_prop, y_prop)
    fakedss.append(fakeds)

  for i, fakeds in enumerate(fakedss):
    shuffle = (i == 0)
    batch_size = int(bs * len(fakeds)) if i == 0 else int(2.5 * bs * len(fakeds))
    
    loader = torch.utils.data.DataLoader(
        fakeds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )
    loaders.append(loader)


  return loaders





#_________________MODEL____________________
#__________________________________________


class DynamicNet(nn.Module):
  def __init__(self, inputshape, outputshape, array, device):
    super().__init__()

    self.layers = nn.Sequential()
    self.act = nn.LeakyReLU()

    prev_num = inputshape
    for i in range(len(array)):
      self.layers.append(nn.Linear(prev_num, array[i]))
      self.layers.append(self.act)
      prev_num = array[i]
    

    self.layers.append(nn.Linear(array[-1], outputshape))
    self.layers.append(nn.Softmax())

    self.to(device)
      

    
  def forward(self, x):
    x = self.layers(x)
    return x

#model = net([10,12])
#print(model(torch.randn(10,12)))







#____________________LEARNING________________________
#____________________________________________________



def learning(trloader, vlloader, criterion, model, optimizer, eps, device = 'cpu', sch = None, print_loss = 1):
    loaders = {"train": trloader, "valid": vlloader}
    avg_losses = {"train": [], "valid": []}
    for epoch in range(eps):
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
                    #if not sch is None:
                    #  sch.step()

                if k == 'valid':
                    model.eval()
                    pred = model(batch[0].to(device))
                    loss = criterion(pred, batch[1].to(device))
                    total_loss += loss.cpu().item()

            if not sch is None:
                sch.step()
            average_loss = total_loss / len(loaders[k])
            avg_losses[k].append(average_loss)
            if(print_loss):
              print(f"pred: {pred[0].cpu().detach().numpy().round(3)}, loss для {k}: {average_loss}")
              print(f"true: {batch[1][0].cpu().numpy().round(3)}")
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


def predict(model,loader, device):
  model.eval()
  preds = []
  for batch in loader:
    pred = model(batch[0].to(device))
    preds.append(pred.cpu())
  preds = torch.cat((preds),0)
  return preds



def pred_results(model, trloader0, y, device, scaler2 = 0, print_loss = 1):
  preds_tr = predict(model,trloader0, device)

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

  rmse = sklearn.metrics.root_mean_squared_error
  rmse0 = rmse(y1_tr,preds1_tr)
  if print_loss:
    print(f'rmse test: {rmse0}')
  return preds_tr, preds1_tr, rmse0
