
x_orig = data.iloc[:,:4].to_numpy()
y_orig = data.iloc[:,5].to_numpy().reshape(-1,1)

tr,vl,ts = split_train(x_orig)
#tr,vl,ts = special_split(x_orig)
x0,y0,scaler1,scaler2 = scale(x_orig, y_orig, scale_data = 1, scale_data_x= x_orig[tr], scale_data_y = y_orig[tr])
#x,y = outs.outs_with_iso(x0, y0)
x = x0
y = y0
#x = x_orig
#y = y_orig

#tr,vl,ts = split_train(x)
#tr0,vl,ts = special_split(x)
#tr = shuffle(tr0)
trloader, trloader0, vlloader, tsloader = loaders(x,y,tr,vl,ts, bs = 0.02)
for batch in trloader:
  print(batch[0].shape)
  print(len(trloader))
  print(x[tr].shape)
  break

model = net().to(device)

opt = torch.optim.AdamW(model.parameters(),lr = 0.1)
crit = torch.nn.MSELoss()
eps = 200
gamma = (0.0001/0.1)**(1/eps)
sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma, last_epoch=-1, verbose='deprecated')
losses = learning(trloader, vlloader, crit, model, opt, eps, device = device, sch = sch)

err_plot(losses)
#save_to_drive(model, 'model_t2', f'/content/drive/MyDrive/Colab Notebooks/диплом/New try/модели/integral/')

preds_ts, preds1_ts =pred_results(model, tsloader, y[ts],device, scaler2 = scaler2)
#analyze_residuals(y[ts], preds_ts)



# x_i, y_i, x_pred_i, preds_i, ids21 = special_preds(model, x[ts], y[ts], device, n_graphs=12)

# x_i1 = [descale(obj, scaler1, col = 4) for obj in x_i]
# y_i1 = [descale(obj, scaler2) for obj in y_i]
# x_pred_i1 = [descale(obj, scaler1, col = 4) for obj in x_pred_i]
# preds_i1 = [descale(obj, scaler2) for obj in preds_i]

# special_plot(x_i1, y_i1, x_pred_i1, preds_i1, ids21, n_cols=3)
