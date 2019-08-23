import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import random
import time
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

path = os.getcwd()
WEIGHTS_PATH = "/tmp/weights.h5"

# LOAD MODEL FROM FILE
model = keras.Sequential([
    keras.layers.Conv1D(31, 250, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear')
])
optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0001, decay=0.0, amsgrad=False)
# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=optimizer,
              loss='mean_squared_error')

model.load_weights(path + WEIGHTS_PATH)

def generateProfile(data, predictions, plt, n, title):
  profile = data[0]
  pred_array = np.zeros(25000)
  for pred in predictions[0]:
    if (pred < 25000 and pred >= 0):
      pred_array[int(round(pred))] = 0.1
      
  profile = np.hstack((profile, pred_array.reshape(25000,1)))

  plt.subplot(n)
  plt.plot(profile[:,0], profile[:,1], linewidth=1, color='black', alpha=0.25,)
  plt.plot(profile[:,0], profile[:,2], linewidth=1, color='lime', alpha=0.25)
  plt.ylabel('RFU')
  plt.title(title)

def makePredictions(prediction_data_path, plt, n, title):
  with open(prediction_data_path, 'r') as file:
    pred_data = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2))

  # select columns of interest: RFU and time
  # pred_data = pred_data[:,1:3].astype(float)

  # normalize
  # pred_data = np.divide(pred_data, 25000)
    pred_data = np.expand_dims(pred_data, axis=0)

    predictions = model.predict(pred_data)
    print(predictions)

    generateProfile(pred_data, predictions, plt, n, title)

def generateProfiles():
  plt.figure(figsize=(30,5))
  makePredictions(path + '/Test/Data/999.txt', plt, 131, 'Sample from normal test set')
  makePredictions(path + '/Test/DataNoDrop/0.txt', plt, 132, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoHarm/0.txt', plt, 133, 'Sample from no-harmonica test set')
  plt.savefig(path + '/plot.png')

generateProfiles()