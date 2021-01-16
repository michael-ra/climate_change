import pandas as pd
import numpy as np

counts_dat = pd.read_csv('./data/fire_precip_temp.csv')
counts_array = counts_dat.to_numpy()

burned_cells = counts_dat['BurnedCells'][:]
land_cells = counts_dat['LandCells'][:]
precip_std = counts_dat['precip_std'][:].to_numpy()
precip_mean = counts_dat['precip_mean'][:].to_numpy()
temp_std = counts_dat['temp_std'][:].to_numpy()
temp_mean = counts_dat['temp_mean'][:].to_numpy()

burned_cells_array = burned_cells.to_numpy()
land_cells_array = land_cells.to_numpy()

shape = burned_cells_array.shape
burned_cleaned = []
precip_means = []
precip_stds =[]
temp_means = []
temp_stds =[]
land_cells = []
for i in range(shape[0]):
    if land_cells_array[i] > 10: 
        if not (np.isnan(burned_cells_array[i]) or np.isnan(precip_std[i]) or np.isnan(precip_mean[i]) or np.isnan(temp_std[i]) or np.isnan(temp_mean[i])):
            burned_cleaned.append(burned_cells_array[i])
            precip_stds.append(precip_std[i])
            precip_means.append(precip_mean[i])
            temp_stds.append(temp_std[i])
            temp_means.append(temp_mean[i])
            land_cells.append(land_cells_array[i])

np.save('./data/fire_counts',np.array(burned_cleaned))
np.save('./data/precip_mean',np.array(precip_means))
np.save('./data/precip_std',np.array(precip_stds))
np.save('./data/temp_mean',np.array(temp_means))
np.save('./data/temp_std',np.array(temp_stds))
np.save('./data/land_cells',np.array(land_cells))