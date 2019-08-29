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
from tensorflow.keras.models import load_model

path = os.getcwd()
MODEL_PATH = "/experiments/TimingTest/model.h5"
WEIGHTS_PATH = "/experiments/checkpoint_weights.hdf5"

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
#
# model = load_model(path + MODEL_PATH)

def generateProfile(data, predictions, plt, n1, n2, n3, title):
  profile = data[0]
  pred_array = np.zeros(25000)
  for pred in predictions[0]:
    if (pred < 25000 and pred >= 0):
      pred_array[int(round(pred))] = 0.1
      
  profile = np.hstack((profile, pred_array.reshape(25000,1)))

  plt.subplot(n1, n2, n3)
  plt.plot(profile[:,0], profile[:,1], linewidth=1, color='black', alpha=0.25,)
  plt.plot(profile[:,0], profile[:,2], linewidth=1, color='lime', alpha=0.25)
  plt.ylabel('RFU')
  plt.title(title)

def makePredictions(prediction_data_path, plt, n1, n2, n3, title):
  with open(prediction_data_path, 'r') as file:
    pred_data = np.zeros((1, 25000, 3))
    pred_data[0] = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
    pred_data = pred_data[:,:,:2]

    # print(pred_data.shape, pred_data)


  # select columns of interest: RFU and time
  # pred_data = pred_data[:,1:3].astype(float)

  # normalize
    pred_data = np.divide(pred_data, 25000)

    # pred_data = np.expand_dims(pred_data, axis=0)
    # print(pred_data.shape, pred_data)
    
    predictions = model.predict(pred_data)
    # print(predictions)

    generateProfile(pred_data, predictions, plt, n1, n2, n3, title)

def generateProfiles():
  plt.figure(figsize=(30,15))
  plt.suptitle("Test set profiles", fontsize=16)
  makePredictions(path + '/Test/Data/999.txt', plt, 3,5,1, 'Sample from normal test set')
  makePredictions(path + '/Test/Data/979.txt', plt, 3,5,2, 'Sample from normal test set')
  makePredictions(path + '/Test/Data/959.txt', plt, 3,5,3, 'Sample from normal test set')
  makePredictions(path + '/Test/Data/939.txt', plt, 3,5,4, 'Sample from normal test set')
  makePredictions(path + '/Test/Data/919.txt', plt, 3,5,5, 'Sample from normal test set')
  makePredictions(path + '/Test/DataNoDrop/1000.txt', plt, 3,5,6, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoDrop/1010.txt', plt, 3,5,7, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoDrop/1020.txt', plt, 3,5,8, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoDrop/1030.txt', plt, 3,5,9, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoDrop/1040.txt', plt, 3,5,10, 'Sample from no-drop test set')
  makePredictions(path + '/Test/DataNoHarm/1000.txt', plt, 3,5,11, 'Sample from no-harmonica test set')
  makePredictions(path + '/Test/DataNoHarm/1010.txt', plt, 3,5,12, 'Sample from no-harmonica test set')
  makePredictions(path + '/Test/DataNoHarm/1020.txt', plt, 3,5,13, 'Sample from no-harmonica test set')
  makePredictions(path + '/Test/DataNoHarm/1030.txt', plt, 3,5,14, 'Sample from no-harmonica test set')
  makePredictions(path + '/Test/DataNoHarm/1040.txt', plt, 3,5,15, 'Sample from no-harmonica test set')
  plt.savefig(path + '/plot-test.png')

  plt.figure(figsize=(30,15))
  plt.suptitle("Training set profiles", fontsize=16)
  makePredictions(path + '/Data/999.txt', plt, 3, 5, 1, 'Sample from normal test set')
  makePredictions(path + '/Data/799.txt', plt, 3, 5, 2, 'Sample from normal test set')
  makePredictions(path + '/Data/599.txt', plt, 3, 5, 3, 'Sample from normal test set')
  makePredictions(path + '/Data/399.txt', plt, 3, 5, 4, 'Sample from normal test set')
  makePredictions(path + '/Data/299.txt', plt, 3, 5, 5, 'Sample from normal test set')
  makePredictions(path + '/DataNoDrop/0.txt', plt, 3, 5, 6, 'Sample from no-drop test set')
  makePredictions(path + '/DataNoDrop/200.txt', plt, 3, 5, 7, 'Sample from no-drop test set')
  makePredictions(path + '/DataNoDrop/400.txt', plt, 3, 5, 8, 'Sample from no-drop test set')
  makePredictions(path + '/DataNoDrop/600.txt', plt, 3, 5, 9, 'Sample from no-drop test set')
  makePredictions(path + '/DataNoDrop/800.txt', plt, 3, 5, 10, 'Sample from no-drop test set')
  makePredictions(path + '/DataNoHarm/0.txt', plt, 3, 5, 11, 'Sample from no-harmonica test set')
  makePredictions(path + '/DataNoHarm/200.txt', plt, 3, 5, 12, 'Sample from no-harmonica test set')
  makePredictions(path + '/DataNoHarm/400.txt', plt, 3, 5, 13, 'Sample from no-harmonica test set')
  makePredictions(path + '/DataNoHarm/600.txt', plt, 3, 5, 14, 'Sample from no-harmonica test set')
  makePredictions(path + '/DataNoHarm/800.txt', plt, 3, 5, 15, 'Sample from no-harmonica test set')
  plt.savefig(path + '/plot-train.png')

generateProfiles()