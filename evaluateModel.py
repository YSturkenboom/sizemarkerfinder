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

WEIGHTS_PATH = "/tmp/weights.h5"

# LOAD MODEL FROM FILE
model = create_model()
model.load_weights(WEIGHTS_PATH)

def generateProfile(data, predictions, plt, n, title):
  profile = data[0]
  pred_array = np.zeros(25000)
  for pred in predictions[0]:
    if (pred < 25000):
      pred_array[int(round(pred))] = 0.1
      
  profile = np.hstack((profile, pred_array.reshape(25000,1)))

  plt.subplot(n)
  plt.plot(profile[:,0], profile[:,1], linewidth=1, color='black', alpha=0.25,)
  plt.plot(profile[:,0], profile[:,2], linewidth=1, color='lime', alpha=0.25);
  plt.ylabel('RFU')
  plt.title(title)

def makePredictions(prediction_data_path, plt, n, title):
  pred_data = []
  with open(prediction_data_path, 'r') as file:
    pred_data = np.array(list(csv.reader(file))[1:])

  # select columns of interest: RFU and time
  pred_data = pred_data[:,1:3].astype(float)

  # normalize
  pred_data = np.divide(pred_data, 25000)
  pred_data = np.expand_dims(pred_data, axis=0)

  predictions = model.predict(pred_data)

  generateProfile(pred_data, predictions, plt, n, title)

def generateProfiles():
  plt.figure(figsize=(30,5))
  makePredictions(path + '/TestData/1.txt', plt, 131, 'Example profile 1')
  makePredictions(path + '/TestData/2.txt', plt, 132, 'Example profile 2')
  makePredictions(path + '/TestData/3.txt', plt, 133, 'Example profile 3')
  plt.show()

generateProfiles()