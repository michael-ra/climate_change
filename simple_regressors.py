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

input_dim = 21 #maybe more later

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
    for j in range(16):
        X[i,5+j]=ltc[j,i]
    #X[i,21] = v_coord[i]
    #X[i,22] = h_coord[i]
    #X[i,5] = p_m_past_month[i]
    #X[i,6] = p_s_past_month[i]
    #X[i,5] = t_s_past_month[i]
    #X[i,8] = t_s_past_month[i]
    

# X will have dim. #data points x input_dim, y #data_points
print(n_points)
#scaler = MinMaxScaler().fit(y)
#y = scaler.transform(y.reshape(1,-1))
#y = np.interp(y, (y.min(), y.max()), (0, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=5/20, random_state=42, shuffle=False)

if regressor == 'rf':
    regr = RandomForestRegressor()
elif regressor == 'lr':
    regr = LinearRegression()
elif regressor == 'gbt':
    regr = GradientBoostingRegressor()

print('Training Regressor...')
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
regr.fit(X_train,y_train)

#save regressor
print('Saving regressor...')
filename = './data/models/'+regressor+'.sav'
pickle.dump(regr, open(filename, 'wb'))
#save regressor features
#if regressor == 'rf':
 #   np.save('./data/models/regressor_feats/feat_imp_'+regressor,regr.feature_importances_)

#make prediction 
print('Predicting on test set...')
prediction = regr.predict(X_test)

#print scores
print('R^2 score is ', regr.score(X_test,y_test))
print('RMSE is ', np.sqrt(sklearn.metrics.mean_squared_error(prediction,y_test)))



