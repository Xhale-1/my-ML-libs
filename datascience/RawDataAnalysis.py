

def irreducible_error(DATA1, y, num_bins=5, epsilon=0.4):
    
      if not isinstance(DATA1, torch.Tensor):
        DATA1 = torch.tensor(DATA1, dtype=torch.float32)
      if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    
      # Создание корзин
      num_bins = num_bins # Увеличим число корзин
      bins_list = [torch.linspace(DATA1[:, i].min(), DATA1[:, i].max(), num_bins + 1)
                  for i in range(DATA1.shape[1])]
      bin_inds = torch.zeros_like(DATA1, dtype=torch.long)
      for i in range(DATA1.shape[1]):
          bin_inds[:, i] = torch.bucketize(DATA1[:, i], bins_list[i], right=True)
      linear_inds = torch.sum(bin_inds * (num_bins ** torch.arange(DATA1.shape[1])), dim=1)
    
      # fig1 = plt.figure()
      # plt.scatter(x[:,0],x[:,1], c = linear_inds)
      # fig1.show()
    
      # Обработка корзин
      irr_err = torch.zeros_like(linear_inds, dtype=torch.float32)
      irr_list, true_irr_list , id_bin_list, bin_pairs_list = [], [], [], []
      outbins = 0
      unique_inds = torch.unique(linear_inds)
    
      for i in unique_inds:
          mask0 = (linear_inds == i)
          dist_matrix = torch.cdist(DATA1[mask0], DATA1[mask0])
          epsilon = epsilon  # Подбираем меньшее значение
          mask = (dist_matrix < epsilon) & (dist_matrix > 0)
          idx_i, idx_j = torch.where(mask)
    
          # print(f"Bin {i}: {len(idx_i)} pairs")
    
          if len(idx_i) < 1000:
            outbins += 1
    
          if len(idx_i) > 1000:  # Минимум 5 пар для надёжности
            diffy = y[mask0][idx_i] - y[mask0][idx_j]
            var_delta_y = torch.var(diffy)
            print(var_delta_y)
            estimated_noise_var = var_delta_y / 2
            irr_err[mask0] = estimated_noise_var
            irr_list.append(torch.tensor([torch.sqrt(estimated_noise_var)]))
            #true_irr_list.append(torch.tensor([sigma(x[mask0]).mean() ** 2]))
            id_bin_list.append(torch.tensor([i]))
            bin_pairs_list.append(torch.tensor([len(idx_i)]))
    
      # Результаты
      irr_tens = torch.stack(irr_list)
      #true_irr_tens = torch.stack(true_irr_list)
      id_bin_tens = torch.stack(id_bin_list)
      bin_pairs_tens = torch.stack(bin_pairs_list)
    
      irr_mean = torch.mean(irr_tens)
      irr_sigma = torch.sqrt(torch.var(irr_tens))
      lower_bound = irr_mean - 3 * irr_sigma
      upper_bound = irr_mean + 3 * irr_sigma
    
      ids = torch.where((irr_tens >= lower_bound) & (irr_tens <= upper_bound))[0]
    
      print(f'Number of all bins: {len(unique_inds)}')
      print(f'Number of bins with zero spsilon close elements: {outbins}')
      print(f"Number of bins outliers: {len(irr_tens) - len(ids)}")
      print(f"Number of non-outliers: {len(ids)}")
    
      result = torch.cat((id_bin_tens[ids], bin_pairs_tens[ids], irr_tens[ids]), dim=1)
      print(result)
      print(f' irr error: {(torch.abs(result[:,2]).mean())}')
