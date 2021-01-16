import pandas as pd
import numpy as np

counts_dat = pd.read_csv('./data/fire_precip_temp.csv')
counts_array = counts_dat.to_numpy()
counts_dat.sort_values('year')
counts_dat.sort_values('month')
year_vals=counts_dat['year'][:].to_numpy()
burned_cells = counts_dat['BurnedCells'][:]
land_cells = counts_dat['LandCells'][:]
precip_std = counts_dat['precip_std'][:].to_numpy()
precip_mean = counts_dat['precip_mean'][:].to_numpy()
temp_std = counts_dat['temp_std'][:].to_numpy()
temp_mean = counts_dat['temp_mean'][:].to_numpy()
v = counts_dat['v'][:].to_numpy()
h = counts_dat['h'][:].to_numpy()

burned_cells_array = burned_cells.to_numpy()
land_cells_array = land_cells.to_numpy()

shape = burned_cells_array.shape
burned_cleaned = []
precip_means = []
precip_stds =[]
temp_means = []
temp_stds =[]
land_cells = []
v_values = []
h_values = []
t_mean_past_month = []
t_std_past_month = []
p_mean_past_month = []
p_std_past_month = []

years=[]

for i in range(1,shape[0]):
    if land_cells_array[i] > 10: 
        if not (np.isnan(burned_cells_array[i]) or np.isnan(precip_std[i]) or np.isnan(precip_mean[i]) or np.isnan(temp_std[i]) or np.isnan(temp_mean[i]) or np.isnan(precip_std[i-1]) or np.isnan(precip_mean[i-1]) or np.isnan(temp_std[i-1]) or np.isnan(temp_mean[i-1])):
            burned_cleaned.append(burned_cells_array[i])
            precip_stds.append(precip_std[i])
            precip_means.append(precip_mean[i])
            temp_stds.append(temp_std[i])
            temp_means.append(temp_mean[i])
            land_cells.append(land_cells_array[i])
            v_values.append(v[i])
            h_values.append(h[i])
            t_mean_past_month.append(temp_mean[i-1])
            t_std_past_month.append(temp_std[i-1])
            p_mean_past_month.append(precip_mean[i-1])
            p_std_past_month.append(precip_mean[i-1])
            years.append(year_vals[i])
            
ltc = np.zeros((17,len(v_values)))
ltc_dat = pd.read_csv('./data/LCT_counts.csv')
ltc_year = ltc_dat['year'][:].to_numpy()
ltc_v = ltc_dat['v'][:].to_numpy()
ltc_h= ltc_dat['h'][:].to_numpy()
n_pixels = ltc_dat['n_pixels'][:].to_numpy()
ltcs = ltc_dat['lct'][:].to_numpy()
ltc_count= len(n_pixels)
for i in range(len(v_values)):
    for j in range(ltc_count):
        if (ltc_year[j]==years[i] and ltc_v[j]==v_values[i] and ltc_h[j] == h_values[i]):
            for k in range(16):
                if ltcs[j]== k+1:
                    ltc[k,i]=n_pixels[j]
            
    

np.save('./data/fire_counts',np.array(burned_cleaned))
np.save('./data/precip_mean',np.array(precip_means))
np.save('./data/precip_std',np.array(precip_stds))
np.save('./data/temp_mean',np.array(temp_means))
np.save('./data/temp_std',np.array(temp_stds))
np.save('./data/land_cells',np.array(land_cells))
np.save('./data/v_coord',np.array(v_values))
np.save('./data/h_coord',np.array(h_values))
np.save('./data/t_mean_past_month',np.array(t_mean_past_month))
np.save('./data/t_std_past_month',np.array(t_std_past_month))
np.save('./data/p_mean_past_month',np.array(p_mean_past_month))
np.save('./data/p_std_past_month',np.array(p_std_past_month))
np.save('./data/ltc',ltc)
