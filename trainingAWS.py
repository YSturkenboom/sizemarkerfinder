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
experimentName = 'ExpA'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

n_samples = 1000
n_epochs = 1000
model_version = 'v1'
shuffle = True
normalize = True
generate_log = True
checkpoints = False
save_model = True
hyperparams = [n_samples, n_epochs, shuffle, normalize, model_version]
hyperparam_names = ['n_samples', 'n_epochs', 'shuffle', 'normalize', 'model_version']

# v1 conv, flatten, 100 dense, 31 dense
# v2 conv, flatten, 100 dense, 100 dense, 31 dense
os.mkdir(path + '/experiments/' + experimentName)

csv_logger = CSVLogger(path + '/experiments/' + experimentName + '/log.csv')
checkpointer = ModelCheckpoint(filepath=path+'/tmp/weights.hdf5', verbose=1, save_best_only=True, mode='min', monitor='loss')

# LOG OUT HYPERPARAMETERS
with open(path + '/experiments/' + experimentName + '/hyperparams.txt', 'w') as f:
    for item in zip(hyperparam_names, hyperparams):
        f.write(str(item[0]) + ': ' + str(item[1]) +'\n')

print('n_samples', n_samples, 'n_epochs', n_epochs)

def readData(data_path, amount):
    files = [f for f in glob.glob(path + data_path)]
    random.shuffle(files)
    data = np.zeros((amount, 25000, 3))
    labels = np.zeros((amount,))
    print(data_path)
    for idx, f in tqdm(enumerate(files[:amount])):
        with open(f, 'r') as file:
            hmm = time.time()
            # stuff = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            # print('stuff', len(stuff), stuff.size, stuff)
            data[idx] = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            # print('array assignment time pre-alloc size', str(time.time() - hmm))

    data = np.transpose(data)

    for experiment in data:
        if (len(labels) == 0):
            sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')

            # experimental: if missing sizemarkers add -1's
            for i in range(31- len(sizemarker_pos)):
                sizemarker_pos.append(-1)

            # print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

            labels = sizemarker_pos
        else:
            sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')

            # experimental: if missing sizemarkers add -1's
            for i in range(31- len(sizemarker_pos)):
                sizemarker_pos.append(-1)

            # print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

            labels = np.vstack((labels, sizemarker_pos))
    
    return (data, labels)

(data, labels) = readData('/Data/*.txt', n_samples)
print('data', data[0], labels[0])
print('size', data.size, labels.size)
(dataNoDrop, labelsNoDrop) = readData('/DataNoDrop/*.txt', n_samples)
(dataNoHarm, labelsNoharm) = readData('/DataNoHarm/*.txt', n_samples)
training_set = np.concatenate((data, dataNoDrop, dataNoHarm))
training_labels = np.concatenate((labels, labelsNoDrop, labelsNoharm))
print('size', training_set.size, training_labels.size)

# select columns of interest: RFU and time
training_set = np.transpose(training_set[:,1:3], (0, 2, 1))

print('size', data.size, labels.size)

(test_set, test_labels) = readData('/Test/*/*.txt', 10)

print('size', test_set.size, test_labels.size)

# select columns of interest: RFU and time
test_set = np.transpose(test_set[:,1:3], (0, 2, 1))

model = keras.Sequential([
    keras.layers.Conv1D(31, 250, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear')
])
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0000001, decay=0.0, amsgrad=False)
model.compile(optimizer=adam,
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

model.save("model_{}_{}.h5".format(n_samples,n_epochs))
print('Test loss', test_loss)

if (save_model):
  model.save_weights(path + '/tmp/weights.h5')
