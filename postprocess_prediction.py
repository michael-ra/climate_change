import pandas as pd
import numpy as np

way = 'ssp126'
mdl='gfdl'

prediction = np.load('./data/predi/prediction'+way+mdl+'.npy')
v = np.load('./data/predi/v_coord'+mdl+way+'.npy')
h = np.load('./data/predi/h_coord'+mdl+way+'.npy')
month = np.load('./data/predi/month'+mdl+way+'.npy')
year = np.load('./data/predi/years'+mdl+way+'.npy')

n =len(h)
numpy_array = np.zeros((n,5))
numpy_array[:,0]=h
numpy_array[:,1]=v
numpy_array[:,2]=prediction
numpy_array[:,3]=month
numpy_array[:,4]=year


df = pd.DataFrame(numpy_array, index=range(n), columns=["h","v","BurnedCells","month","year"])

df.to_csv('cmip6_'+way+mdl+'_firecounts_predicted.csv', encoding='utf-8', index=False)