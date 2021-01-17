import pandas as pd
import numpy as np
from tqdm import tqdm

#cmip data
mdl = 'gfdl'
way = 'ssp370'
cmip_dat = pd.read_csv('./datasets/cmip6_ssp370_gfdl-esm4.csv')
cmip_dat.sort_values(['month','year'],ascending=[True, True], inplace=True)
cmip_dat.sort_values(['v','h'],ascending=[True, True], inplace=True)
cmip_year_vals=cmip_dat['year'][:].to_numpy()
cmip_month=cmip_dat['month'][:].to_numpy()
precip_std = cmip_dat['precip_std'][:].to_numpy()
precip_mean = cmip_dat['precip_mean'][:].to_numpy()
temp_std = cmip_dat['t2m_std'][:].to_numpy()
temp_mean = cmip_dat['t2m_mean'][:].to_numpy()
cmip_v = cmip_dat['v'][:].to_numpy()
cmip_h = cmip_dat['h'][:].to_numpy()

#landcells data
counts_dat = pd.read_csv('./data/fire_precip_temp.csv')
land_cells = counts_dat['LandCells'][:]
land_cells_array = land_cells.to_numpy()
v = counts_dat['v'][:].to_numpy()
h = counts_dat['h'][:].to_numpy()


precip_means = []
precip_stds =[]
temp_means = []
temp_stds =[]
years=[]
months=[]
t_mean_past_month = []
t_std_past_month = []
p_mean_past_month = []
p_std_past_month = []

land_cells = []
v_values = []
h_values = []




def get_n_land_cells(v_,h_):
    n = 0
    for i in range(len(v)):
        if v[i]==v_ and h[i]==h_:
            return land_cells_array[i]
    print('warning')  
    return n

for i in tqdm(range(len(cmip_v))):
    n_land_cells = get_n_land_cells(cmip_v[i],cmip_h[i])
    if n_land_cells>10: 
        if not ( np.isnan(precip_std[i]) or np.isnan(precip_mean[i]) or np.isnan(temp_std[i]) or np.isnan(temp_mean[i])):
            precip_stds.append(precip_std[i])
            precip_means.append(precip_mean[i])
            temp_stds.append(temp_std[i])
            temp_means.append(temp_mean[i])
            land_cells.append(n_land_cells)
            v_values.append(cmip_v[i])
            h_values.append(cmip_h[i])
            years.append(cmip_year_vals[i])
            months.append(cmip_month[i])
            t_mean_past_month.append(temp_mean[i-1])
            t_std_past_month.append(temp_std[i-1])
            p_mean_past_month.append(precip_mean[i-1])
            p_std_past_month.append(precip_mean[i-1])
            
            


ltc = np.zeros((17,len(v_values)))
ltc_dat = pd.read_csv('./data/LCT_counts.csv')
ltc_year = ltc_dat['year'][:].to_numpy()
ltc_v = ltc_dat['v'][:].to_numpy()
ltc_h= ltc_dat['h'][:].to_numpy()
n_pixels = ltc_dat['n_pixels'][:].to_numpy()
ltcs = ltc_dat['lct'][:].to_numpy()
ltc_count= len(n_pixels)

for i in tqdm(range(len(v_values))):
    for j in range(ltc_count):
        if (ltc_year[j]==2017 and ltc_v[j]==v_values[i] and ltc_h[j] == h_values[i]):
            for k in range(16):
                if ltcs[j]== k+1:
                    ltc[k,i]=n_pixels[j]
                    
#np.save('./data/fire_counts',np.array(burned_cleaned))
np.save('./data/predi/precip_mean'+mdl+way,np.array(precip_means))
np.save('./data/predi/precip_std'+mdl+way,np.array(precip_stds))
np.save('./data/predi/temp_mean'+mdl+way,np.array(temp_means))
np.save('./data/predi/temp_std'+mdl+way,np.array(temp_stds))
np.save('./data/predi/land_cells'+mdl+way,np.array(land_cells))
np.save('./data/predi/v_coord'+mdl+way,np.array(v_values))
np.save('./data/predi/h_coord'+mdl+way,np.array(h_values))
np.save('./data/predi/ltc'+mdl+way,ltc)
np.save('./data/predi/month'+mdl+way,months)
np.save('./data/predi/years'+mdl+way,years)
np.save('./data/predi/t_mean_past_month'+mdl+way,np.array(t_mean_past_month))
np.save('./data/predi/t_std_past_month'+mdl+way,np.array(t_std_past_month))
np.save('./data/predi/p_mean_past_month'+mdl+way,np.array(p_mean_past_month))
np.save('./data/predi/p_std_past_month'+mdl+way,np.array(p_std_past_month))