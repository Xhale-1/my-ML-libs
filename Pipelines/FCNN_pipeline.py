
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import sklearn
import shutil






def split_train(x00):
  ids = np.random.choice(len(x00), len(x00), replace=False)
  tr,vl,ts = np.split(ids, [int(0.63 * len(x00)), int(0.79 * len(x00)) ])
  return tr,vl,ts



def scale(x00,y00, scale_data = 0, scale_data_x = 0, scale_data_y = 0 ):

  if not isinstance(x00, np.ndarray):
    x00 = np.array(x00)
  if not isinstance(y00, np.ndarray):
    y00 = np.array(y00)
  if not isinstance(scale_data_x, np.ndarray):
    x00 = np.array(scale_data_x)
  if not isinstance(scale_data_y, np.ndarray):
    y00 = np.array(scale_data_y)

  scaler1 = StandardScaler()
  scaler2 = StandardScaler()

  DATAX = x00
  DATAY = y00
  if scale_data:
    DATAX = scale_data_x
    DATAY = scale_data_y

  print(DATAX[:5])
  scaler1.fit(DATAX)
  scaler2.fit(DATAY)
  x0 = scaler1.transform(x00)
  y0 = scaler2.transform(y00)

  x = torch.tensor(x0, dtype = torch.float32)
  y = torch.tensor(y0, dtype = torch.float32).reshape(-1,1)

  return x,y,scaler1,scaler2




def loaders(x,y,tr,vl,ts):
  trfakeds = list(zip(x[tr],y[tr]))
  vlfakeds = list(zip(x[vl],y[vl]))
  tsfakeds = list(zip(x[ts],y[ts]))

  trloader = torch.utils.data.DataLoader(trfakeds, batch_size= int(0.01*len(trfakeds)), shuffle = True, num_workers=2)
  trloader0 = torch.utils.data.DataLoader(trfakeds, batch_size= int(0.01*len(trfakeds)), shuffle = False, num_workers=2)
  vlloader = torch.utils.data.DataLoader(vlfakeds, batch_size= int(0.1*len(vlfakeds)), shuffle = False, num_workers=2)
  tsloader = torch.utils.data.DataLoader(tsfakeds, batch_size= int(0.1*len(tsfakeds)), shuffle = False, num_workers=2)

  return trloader, trloader0, vlloader, tsloader




def learning(trloader, vlloader, criterion, model, optimizer, eps, device = 'cpu', sch = None):
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

                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if k == 'valid':
                    model.eval()
                    pred = model(batch[0].to(device))
                    loss = criterion(pred, batch[1].to(device))
                    total_loss += loss.item()

            if not sch is None:
              sch.step()
            average_loss = total_loss / len(loaders[k])
            avg_losses[k].append(average_loss)
            print(f"pred: {pred[0].detach().numpy().round(3)}, loss для {k}: {average_loss}")
            print(f"pred: {batch[1][0].numpy().round(3)}")
    return avg_losses



def err_plot(losses):
  plt.plot(losses['train'], c = 'red')
  plt.plot(losses['valid'])
  plt.yscale('log')
  plt.show()



def save_to_drive(model,name, path):
  torch.save(model.state_dict(), f"/content/{name}.pth")
  shutil.copy(f"/content/{name}.pth", path)


def predict(model,loader, device = 'cpu'):
  model.eval()
  preds = []
  for batch in loader:
    pred = model(batch[0].to(device))
    preds.append(pred.cpu())
  preds = torch.cat((preds),0)
  return preds


def pred_results(model, trloader0, y, scaler2):
  preds_tr = predict(model,trloader0)

  preds1_tr = scaler2.inverse_transform(preds_tr.detach().numpy())
  y1_tr = scaler2.inverse_transform(y.detach().numpy())

  print(np.array(list(zip(preds1_tr[:5],y1_tr[:5]))).reshape(-1,2))

  rmse = sklearn.metrics.root_mean_squared_error
  print(f'rmse train: {rmse(y1_tr,preds1_tr)}')
  return preds_tr, preds1_tr
