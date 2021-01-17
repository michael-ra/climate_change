from google.colab import files
uploaded_files = files.upload()

import pandas as pd
climate_data_samp = pd.read_csv("/content/drive/MyDrive/fire_precip_temp.csv")

climate_data_samp_grouped = climate_data_samp.groupby(by=['v', 'h'], dropna=False)

pd.options.display.max_columns = None
pd.options.display.max_rows = None

listOfCoordsData = []
listOfLabels = []
condDataV = []
condDataH = []
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  climate_data_samp_grouped.groups.keys()
  for k in climate_data_samp_grouped.groups.keys():
    listOfCoordsData.append(climate_data_samp_grouped.get_group(k))

#print(next(iter(listOfCoordsData), None))
#for df in listOfCoordsData:
# df.sort_values(['year', 'month'])


listOfCoordsData[:] = [df.dropna(subset=['precip_std']).dropna(subset=['precip_mean']).dropna(subset=['temp_mean']).dropna(subset=['temp_std']) for df in listOfCoordsData]

for df in listOfCoordsData:
  condDataV.append(df['v'])
  condDataH.append(df['h'])

listOfCoordsData[:] = [df.sort_values(['year', 'month']).drop(['fname','dim_0','v','h'], axis=1) for df in listOfCoordsData]

listOfCoordsData[:] = [df.drop(['year','month','ProductEndDay','ProductStartDay'], axis=1) for df in listOfCoordsData]

for df in listOfCoordsData:
  listOfLabels.append(df['BurnedCells'])

listOfCoordsData[:] = [df.drop(['BurnedCells'], axis=1) for df in listOfCoordsData]

print(listOfCoordsData)


import tensorflow as tf

yIn = np.array(listOfLabels, dtype=object)
inputTimeNP = [x.to_numpy() for x in listOfCoordsData]
inputTimeNP = np.array(inputTimeNP, dtype=object)

print(inputTimeNP.shape)
print(yIn.shape)

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


model = tf.keras.Sequential([
   tf.keras.layers.GRU(120, return_sequences=True, input_shape=(None,7)),
   tf.keras.layers.GRU(80, return_sequences=True),
   tf.keras.layers.GRU(20, return_sequences=True),
   tf.keras.layers.GRU(30, return_sequences=True),
   tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])

model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
model.summary()

#DO NOT USE-PER EXAMPLE TRAINING
#input
for index, elem in enumerate(inputTimeNP):
        input = elem
        yIn = np.array(listOfLabels[index])
        o = input.shape
        input = input.reshape(1,o[0],7)
        print(input.shape)
        print(yIn.shape)
        yIn = yIn.reshape(1,o[0])
        print(yIn.shape)
        model.fit(x=input, y=yIn, epochs=5)
        #conditional?, inCondV, inCondH)
