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
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LambdaCallback

path = os.getcwd()

# VARIABLES AND HYPERPARAMETERS
experimentName = 'Aug29'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

n_samples = 10
n_epochs = 10
model_version = 'v1'
shuffle = True
normalize = True
generate_log = True
checkpoints = True
save_model = True
create_plots_at_epochs = [1,5,10,50,100,200,300,400,500,600,700,800,900,999]
hyperparams = [n_samples, n_epochs, shuffle, normalize, model_version, create_plots_at_epochs]
hyperparam_names = ['n_samples', 'n_epochs', 'shuffle', 'normalize', 'model_version', 'create_plots_at_epochs']


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

def makePredictions(current_model, prediction_data_path, plt, n1, n2, n3, title):
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
    
    predictions = current_model.predict(pred_data)
    # print(predictions)

    generateProfile(pred_data, predictions, plt, n1, n2, n3, title)

def generateProfiles(current_model, epoch):
  plt.figure(figsize=(30,15))
  plt.suptitle("Experiment " + experimentName + ": Test set profiles at epoch " + epoch, fontsize=16)
  makePredictions(current_model, path + '/Test/Data/999.txt', plt, 3,5,1, 'Sample from normal test set')
  makePredictions(current_model, path + '/Test/Data/979.txt', plt, 3,5,2, 'Sample from normal test set')
  makePredictions(current_model, path + '/Test/Data/959.txt', plt, 3,5,3, 'Sample from normal test set')
  makePredictions(current_model, path + '/Test/Data/939.txt', plt, 3,5,4, 'Sample from normal test set')
  makePredictions(current_model, path + '/Test/Data/919.txt', plt, 3,5,5, 'Sample from normal test set')
  makePredictions(current_model, path + '/Test/DataNoDrop/1000.txt', plt, 3,5,6, 'Sample from no-drop test set')
  makePredictions(current_model, path + '/Test/DataNoDrop/1010.txt', plt, 3,5,7, 'Sample from no-drop test set')
  makePredictions(current_model, path + '/Test/DataNoDrop/1020.txt', plt, 3,5,8, 'Sample from no-drop test set')
  makePredictions(current_model, path + '/Test/DataNoDrop/1030.txt', plt, 3,5,9, 'Sample from no-drop test set')
  makePredictions(current_model, path + '/Test/DataNoDrop/1040.txt', plt, 3,5,10, 'Sample from no-drop test set')
  makePredictions(current_model, path + '/Test/DataNoHarm/1000.txt', plt, 3,5,11, 'Sample from no-harmonica test set')
  makePredictions(current_model, path + '/Test/DataNoHarm/1010.txt', plt, 3,5,12, 'Sample from no-harmonica test set')
  makePredictions(current_model, path + '/Test/DataNoHarm/1020.txt', plt, 3,5,13, 'Sample from no-harmonica test set')
  makePredictions(current_model, path + '/Test/DataNoHarm/1030.txt', plt, 3,5,14, 'Sample from no-harmonica test set')
  makePredictions(current_model, path + '/Test/DataNoHarm/1040.txt', plt, 3,5,15, 'Sample from no-harmonica test set')
  plt.savefig(path + '/plot-ep'+epoch+'-test.png')

  plt.figure(figsize=(30,15))
  plt.suptitle("Experiment " + experimentName + ": Training set profiles at epoch " + epoch, fontsize=16)
  makePredictions(current_model, path + '/Data/899.txt', plt, 3, 5, 1, 'Sample from normal training set')
  makePredictions(current_model, path + '/Data/699.txt', plt, 3, 5, 2, 'Sample from normal training set')
  makePredictions(current_model, path + '/Data/499.txt', plt, 3, 5, 3, 'Sample from normal training set')
  makePredictions(current_model, path + '/Data/299.txt', plt, 3, 5, 4, 'Sample from normal training set')
  makePredictions(current_model, path + '/Data/90.txt', plt, 3, 5, 5, 'Sample from normal training set')
  makePredictions(current_model, path + '/DataNoDrop/1100.txt', plt, 3, 5, 6, 'Sample from no-drop training set')
  makePredictions(current_model, path + '/DataNoDrop/1200.txt', plt, 3, 5, 7, 'Sample from no-drop training set')
  makePredictions(current_model, path + '/DataNoDrop/1400.txt', plt, 3, 5, 8, 'Sample from no-drop training set')
  makePredictions(current_model, path + '/DataNoDrop/1600.txt', plt, 3, 5, 9, 'Sample from no-drop training set')
  makePredictions(current_model, path + '/DataNoDrop/1800.txt', plt, 3, 5, 10, 'Sample from no-drop training set')
  makePredictions(current_model, path + '/DataNoHarm/1100.txt', plt, 3, 5, 11, 'Sample from no-harmonica training set')
  makePredictions(current_model, path + '/DataNoHarm/1200.txt', plt, 3, 5, 12, 'Sample from no-harmonica training set')
  makePredictions(current_model, path + '/DataNoHarm/1400.txt', plt, 3, 5, 13, 'Sample from no-harmonica training set')
  makePredictions(current_model, path + '/DataNoHarm/1600.txt', plt, 3, 5, 14, 'Sample from no-harmonica training set')
  makePredictions(current_model, path + '/DataNoHarm/1800.txt', plt, 3, 5, 15, 'Sample from no-harmonica training set')
  plt.savefig(path + '/plot-ep'+epoch+'-train.png')

class PlotCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (epoch in create_plots_at_epochs): 
      generateProfiles(self.model, epoch)

# v1 conv, flatten, 100 dense, 31 dense
# v2 conv, flatten, 100 dense, 100 dense, 31 dense
os.mkdir(path + '/experiments/' + experimentName)

csv_logger = CSVLogger(path + '/experiments/' + experimentName + '/log.csv')
checkpointer = ModelCheckpoint(filepath=path+'/experiments/checkpoint_weights.hdf5', verbose=1, save_best_only=True, mode='min', monitor='loss')

# LOG OUT HYPERPARAMETERS
with open(path + '/experiments/' + experimentName + '/hyperparams.txt', 'w') as f:
    for item in zip(hyperparam_names, hyperparams):
        f.write(str(item[0]) + ': ' + str(item[1]) +'\n')

print('n_samples', n_samples, 'n_epochs', n_epochs)

def readData(data_path, amount):
    files = [f for f in glob.glob(path + data_path)]
    random.shuffle(files)
    data = np.zeros((amount, 25000, 3))
    labels = np.zeros((amount, 31))
    print(data_path)
    for idx, f in tqdm(enumerate(files[:amount])):
        with open(f, 'r') as file:
            # stuff = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            # print('stuff', len(stuff), stuff.size, stuff)
            loadeddata =  np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            data[idx] = loadeddata
            
            print('loaded', loadeddata)

            sizemarker_pos = list(int(datapoint[0]) for datapoint in loadeddata if datapoint[2] != -1)
            for _ in range(31- len(sizemarker_pos)):
              sizemarker_pos.append(-1)

            print(sizemarker_pos)

            labels[idx] = sizemarker_pos

            
            # print('array assignment time pre-alloc size', str(time.time() - hmm))
    # data = np.transpose(data)

    # for idx, experiment in tqdm(enumerate(data)):
    #   print(experiment, np.transpose(experiment), np.transpose(experiment)[0])
    #   sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[2] != '-1')
    #   for _ in range(31- len(sizemarker_pos)):
    #     sizemarker_pos.append(-1)
      
    #   labels[idx] = sizemarker_pos
    #   # print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

    return (data[:,:,:2], labels)

(data, labels) = readData('/Data/*.txt', n_samples)
print(data.shape)
print(labels.shape, labels[0])
(dataNoDrop, labelsNoDrop) = readData('/DataNoDrop/*.txt', n_samples)
(dataNoHarm, labelsNoharm) = readData('/DataNoHarm/*.txt', n_samples)
training_set = np.vstack((data, dataNoDrop, dataNoHarm))
training_labels = np.vstack((labels, labelsNoDrop, labelsNoharm))
# print('size', training_set.size, training_labels.size)

# # # select columns of interest: RFU and time
# training_set = np.transpose(training_set[:,1:3], (0, 2, 1))

# print('size', data.size, labels.size)

(test_set, test_labels) = readData('/Test/*/*.txt', n_samples)

print('size', test_set.size, test_labels.size)
print('nan', training_set[0], test_set[0])

if (normalize):
  # normalize: divide RFU by 1000, time by 25000, labels by 1500
  training_set = np.divide(training_set, 25000)
  test_set = np.divide(test_set, 25000)
# select columns of interest: RFU and time
# test_set = np.transpose(test_set[:,1:3], (0, 2, 1))

model = keras.Sequential([
    keras.layers.Conv1D(31, 250, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear')
])
optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0, amsgrad=False)
# optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(optimizer=optimizer,
              loss='mean_squared_error')
              
from keras.utils import plot_model
plot_model(model, to_file=path + "/experiments/" + experimentName +  "/model.png")

callbacks = []
if (checkpoints):
  callbacks.append(checkpointer)
if (generate_log):
  callbacks.append(csv_logger)
if (len(create_plots_at_epochs) > 0):
  plot_callback = PlotCallback()
  callbacks.append(plot_callback)
  
startFit = time.time()
model.fit(training_set, training_labels, validation_split=0.2, shuffle=shuffle, batch_size=32, epochs=n_epochs, callbacks=callbacks)

test_loss = model.evaluate(test_set, test_labels)
with open(path + '/experiments/' + experimentName + '/hyperparams.txt', 'a') as f:
  f.write('model_fit_time: ' + str(time.time() - startFit) + '\n')
  f.write('test_loss: ' + str(test_loss))
print('Test loss', test_loss)


print('Test loss', test_loss)

if (save_model):
  model.save(path + '/experiments/' + experimentName +  "/model.h5")
  # model.save_weights(path + '/experiments/' + experimentName + '/weights.h5')
