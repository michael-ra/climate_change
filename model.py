from google.colab import drive
drive.mount('/content/drive', force_remount=True)
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

#RESIZE
import numpy as np
import tensorflow as tf

yIn = np.array(listOfLabels, dtype=object)
inputTimeNP = [x.to_numpy() for x in listOfCoordsData]
inputTimeNP = np.array(inputTimeNP, dtype=object)

print(inputTimeNP.shape)
print(yIn.shape)


#MODEL
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os

model = tf.keras.Sequential([
   tf.keras.layers.GRU(24, return_sequences=True, input_shape=(None,7)),
   tf.keras.layers.GRU(42, return_sequences=True),
   tf.keras.layers.GRU(128, return_sequences=True),
   tf.keras.layers.GRU(5, return_sequences=True),
   tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
])
opt = tf.keras.optimizers.Adam(
    learning_rate=0.1, beta_1=0.85, beta_2=0.989, epsilon=1e-07, amsgrad=True,
    name='Adam'
)

model.compile(opt, loss='mean_squared_error', metrics=['accuracy', tf.keras.metrics.MeanSquaredError()])
model.summary()

#GENERATOR
def batch_generator_sub_batch(X, Y, batch_size):
    indices = np.arange(len(inputTimeNP)) 
    batch = []
    xAll = []
    yAll = []
    while True:
        np.random.shuffle(indices) 
        for i in indices:
          batch.append(i)
          xIn = inputTimeNP[i]
          yIn = np.array(listOfLabels[i])
          xAll.append(xIn)
          yAll.append(yIn)
          if len(batch)==batch_size:
            xAllnp = np.asarray(xAll, dtype=np.object)
            yAllnp = np.asarray(yAll, dtype=np.object)
            #print(xAllnp.shape)
            xAllnpForSplit = np.array_split(xAllnp[0], 10)  #268/4=67 % 0 (!)<-------<
            #print(xAllnpForSplit)
            #print(yAllnp.shape)
            yAllnpForSplit = np.array_split(xAllnp[0], 10)
            #print(yAllnpForSplit)
            xAllnpForSplit = np.asarray(xAllnpForSplit, dtype=np.object)
            yAllnpForSplit = np.asarray(yAllnpForSplit, dtype=np.object)
            print(xAllnpForSplit.shape)
            print(yAllnpForSplit.shape)
            #reset
            xAll = []
            yAll = []
            xAllnp = []
            yAllnp = []
            batch = []
            yield (tf.ragged.constant(yAllnpForSplit.astype('float32', copy=False)),tf.ragged.constant(yAllnpForSplit.astype('int', copy=False)))
            
#GENERATOR
def batch_generator_main(X, Y, batch_size):
    indices = np.arange(len(inputTimeNP)) 
    batch = []
    xAll = []
    yAll = []
    while True:
        for i in indices:
         if i < len(indices):
          batch.append(i)
          xIn = inputTimeNP[i]
          yIn = np.array(listOfLabels[i])
          xAll.append(xIn)
          yAll.append(yIn)
          if len(batch)==batch_size:
            xAllnp = np.asarray(xAll, dtype=np.object)
            yAllnp = np.asarray(yAll, dtype=np.object)
            print(xAllnp.shape)
            print(yAllnp.shape)
            xAll = []
            yAll = []
            batch = []
            yield (tf.convert_to_tensor(xAllnp.astype('float32')),tf.convert_to_tensor(yAllnp.astype('int')))

#tf.executing_eagerly()
model.fit(batch_generator_main(inputTimeNP, listOfLabels, 1), epochs=1)
#model.predict


#UNUSED
           # o = xAll.shape
           # xIn = xIn.reshape(1,o[0],7)
           # print(input.shape)
           # print(yIn.shape)
           # yIn = yIn.reshape(1,o[0])
          #  print(yIn.shape)
         #   model.fit(x=input, y=yIn, epochs=5)
          #, inCondV, inCondH)
