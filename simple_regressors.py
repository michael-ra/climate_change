print('Loading modules...')
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
import argparse

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--regressor", default='rf', help="choose a regressor, either random forest (rf), linear regression (lr), gradient boosting regressor (gbt)")
args = parser.parse_args()
regressor = args.regressor

input_dim = 4 #maybe more later

#load data
print('Loading data...')


precipitation = np.load('./data/era5_precipitaion_mean_var.npy') #assuming dim n_pointsx2
temperature = np.load('./data/era5_temperature_mean_var.npy') #assuming dim n_pointsx2

fire_counts = np.load(path_to_fire_counts) #assuming dim n_pointsx1
n_points = len(fire_counts)

X = np.zeros((n_points, input_dim))
y = fire_counts

for i in range(n_points):
    X[i,0] = precipitation[i,0]
    X[i,1] = precipitation[i,1]
    X[i,2] = temperature[i,0]
    X[i,3] = temperature[i,1]


# X will have dim. #data points x input_dim, y #data_points

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4/14, random_state=42, shuffle=False)

if regressor == 'rf':
    regr = RandomForestRegressor()
elif regressor == 'lr':
    regr = LinearRegression()
elif regressor == 'gbt':
    regr = GradientBoostingRegressor()

print('Training Regressor...')
regr.fit(X_train,y_train)

#save regressor
print('Saving regressor...')
filename = './models/'+regressor'.sav'
pickle.dump(regr, open(filename, 'wb'))
#save regressor features
if regressor == 'rf':
    np.save('/models/regressor_feats/feat_imp_'+regressor,regr.feature_importances_)

#make prediction 
print('Predicting on test set...')
prediction = regr.predict(X_test)

#print scores
print('R^2 score is ', regr.score(X_test,y_test))
print('RMSE is ', np.sqrt(sklearn.metrics.mean_squared_error(prediction,y_test)))



