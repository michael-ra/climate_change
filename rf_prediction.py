print('Loading modules...')
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
import argparse
import numpy as np

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--regressor", default='rf', help="choose a regressor, either random forest (rf), linear regression (lr), gradient boosting regressor (gbt)")
args = parser.parse_args()
regressor = args.regressor

input_dim = 25 #maybe more later

#load data
print('Loading data...')
mdl = 'gfdl'
way='ssp126'

precip_mean = np.load('./data/predi/precip_mean'+mdl+way+'.npy')
precip_std = np.load('./data/predi/precip_std'+mdl+way+'.npy')
temp_mean = np.load('./data/predi/temp_mean'+mdl+way+'.npy')
temp_std = np.load('./data/predi/temp_std'+mdl+way+'.npy')
land_cells = np.load('./data/predi/land_cells'+mdl+way+'.npy')
v_coord = np.load('./data/predi/v_coord'+mdl+way+'.npy')
h_coord = np.load('./data/predi/h_coord'+mdl+way+'.npy')
ltc = np.load('./data/predi/ltc'+mdl+way+'.npy')
p_m_past_month=np.load('./data/predi/p_mean_past_month'+mdl+way+'.npy')
p_s_past_month=np.load('./data/predi/p_std_past_month'+mdl+way+'.npy')
t_m_past_month=np.load('./data/predi/t_mean_past_month'+mdl+way+'.npy')
t_s_past_month=np.load('./data/predi/t_std_past_month'+mdl+way+'.npy')

n_points = len(v_coord)

X = np.zeros((n_points, input_dim))


print(n_points)
for i in range(n_points):
    X[i,0] = precip_mean[i]
    X[i,1] = precip_std[i]
    X[i,2] = temp_mean[i]
    X[i,3] = temp_std[i]
    X[i,4] = land_cells[i]

    for j in range(16):

        X[i,5+j]=ltc[j,i]
        
    X[i,21] = p_m_past_month[i]
    X[i,22] = p_s_past_month[i]
    X[i,23] = t_m_past_month[i]
    X[i,24] = t_s_past_month[i]



if regressor == 'rf':
    regr = RandomForestRegressor()
elif regressor == 'lr':
    regr = LinearRegression()
elif regressor == 'gbt':
    regr = GradientBoostingRegressor()

print('Loading regressor...')
filename = './data/models/'+regressor+'.sav'
regr=pickle.load( open(filename, 'rb'))

#make prediction 
print('Predicting on test set...')
prediction = regr.predict(X)

np.save('./data/predi/prediction'+way+mdl,prediction)

