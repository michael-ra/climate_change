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
parser.add_argument("--mode",default='val',help="either validation (val) or precidtion (pred) mode")
args = parser.parse_args()
regressor = args.regressor
mode = args.mode

input_dim = 25 

#load data
print('Loading data...')


precip_mean = np.load('./data/precip_mean.npy')
precip_std = np.load('./data/precip_std.npy')
temp_mean = np.load('./data/temp_mean.npy')
temp_std = np.load('./data/temp_std.npy')
land_cells = np.load('./data/land_cells.npy')
v_coord = np.load('./data/v_coord.npy')
h_coord = np.load('./data/h_coord.npy')
p_m_past_month = np.load('./data/p_mean_past_month.npy')
p_s_past_month = np.load('./data/p_std_past_month.npy')
t_m_past_month = np.load('./data/t_mean_past_month.npy')
t_s_past_month = np.load('./data/t_std_past_month.npy')
years = np.load('./data/years.npy')
ltc = np.load('./data/ltc.npy')

fire_counts = np.load('./data/fire_counts.npy') #assuming dim n_pointsx1
n_points = len(fire_counts)

X = np.zeros((n_points, input_dim))
y = fire_counts


for i in range(n_points):
    X[i,0] = precip_mean[i]
    X[i,1] = precip_std[i]
    X[i,2] = temp_mean[i]
    X[i,3] = temp_std[i]
    X[i,4] = land_cells[i]
    kl = 0
    for j in range(16):

        X[i,5+j]=ltc[j,i]

    X[i,21] = p_m_past_month[i]
    X[i,22] = p_s_past_month[i]
    X[i,23] = t_m_past_month[i]
    X[i,24] = t_s_past_month[i]


print(n_points)

if mode == 'val':
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(n_points):
        if years[i]<2015:
            X_train.append(X[i,:])
            y_train.append(y[i])
        else:
            X_test.append(X[i,:])
            y_test.append(y[i])

else:
    X_train = X
    y_train = y
    

if regressor == 'rf':
    regr = RandomForestRegressor()
elif regressor == 'lr':
    regr = LinearRegression()
elif regressor == 'gbt':
    regr = GradientBoostingRegressor()

print('Training Regressor...')

regr.fit(X_train,y_train)

#save regressor
if mode == 'pred':
    print('Saving regressor...')
    filename = './data/models/'+regressor+'.sav'
    pickle.dump(regr, open(filename, 'wb'))
else:
    #make prediction 
    print('Predicting on test set...')
    prediction = regr.predict(X_test)
    print(len(prediction))
    #print scores
    print('R^2 score is ', regr.score(X_test,y_test))
    print('RMSE is ', np.sqrt(sklearn.metrics.mean_squared_error(prediction,y_test)))

    rmse=0
    for i in range(len(y_test)):
        if y_test[i]>0 or prediction[i]>0:
            rmse+=  2*np.abs(prediction[i]-y_test[i])/(y_test[i]+prediction[i])

    rmse *=1/len(y)

    print(rmse)



