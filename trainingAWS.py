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

# VARIABLES AND HYPERPARAMETERS
experimentName = 'Aug29'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

n_samples = 1000
n_epochs = 1000
model_version = 'v1'
shuffle = True
normalize = True
generate_log = True
checkpoints = True
save_model = True
hyperparams = [n_samples, n_epochs, shuffle, normalize, model_version]
hyperparam_names = ['n_samples', 'n_epochs', 'shuffle', 'normalize', 'model_version']

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

callbacks = []
if (checkpoints):
  callbacks.append(checkpointer)
if (generate_log):
  callbacks.append(csv_logger)
  
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
