import sys
import os

# Do not use GPU (use CPU)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import random
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

path = os.getcwd()
print(path)
n_samples = 10
n_epochs = 1000
model_version = 'test'

# v1 conv, flatten, 100 dense, 31 dense
# v2 conv, flatten, 100 dense, 100 dense, 31 dense

csv_logger = CSVLogger(path + '/tmp/v' + str(model_version) + 'e' + str(n_epochs) + 'n' + str(n_samples) + 'log.csv', append=True)
# checkpointer = ModelCheckpoint(filepath=path+'/tmp/weights.hdf5', verbose=1, save_best_only=True, mode='min', monitor='loss')

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))

# read cmd line args
# if (len(sys.argv) > 1):
# 	n_samples = int(sys.argv[1])
# 	if (n_samples > 1000):
# 		print("sample size exceeds available samples")
# 		sys.exit()
# if (len(sys.argv) > 2):
# 	n_epochs  = int(sys.argv[2])

print('n_samples', n_samples, 'n_epochs', n_epochs)

def readData(data_path, amount):
    data = np.array([])

    files = [f for f in glob.glob(path + data_path)]
    random.shuffle(files)
    print(data_path)
    for f in tqdm(files[:amount]):
        with open(f, 'r') as file:
            if (data.size == 0):
                data = np.array(list(csv.reader(file))[1:])
            else:
                data = np.dstack((data, list(csv.reader(file))[1:]))

    data = np.transpose(data)
    labels = np.array([])

    for experiment in data:
        if (len(labels) == 0):
            sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')

            # experimental: if missing sizemarkers add -1's
            for i in range(31- len(sizemarker_pos)):
                sizemarker_pos.append(-1)

    #         print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

            labels = sizemarker_pos
        else:
            sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')

            # experimental: if missing sizemarkers add -1's
            for i in range(31- len(sizemarker_pos)):
                sizemarker_pos.append(-1)

            # print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

            labels = np.vstack((labels, sizemarker_pos))
    
    # select columns of interest: RFU and time
    data = np.transpose(data[:,1:3], (0, 2, 1))
    return (data, labels)

# 80/20 train/test split
# training_set_size = int(round(n_samples * 0.8))

(data, labels) = readData('/Data/*.txt', n_samples)
print('data', data[0])
(dataNoDrop, labelsNoDrop) = readData('/DataNoDrop/*.txt', n_samples)
(dataNoHarm, labelsNoharm) = readData('/DataNoHarm/*.txt', n_samples)
training_set = np.append(data, [dataNoDrop, dataNoHarm])
training_labels = np.append(labels, [labelsNoDrop, labelsNoharm])

print('size', data.size, labels.size)

(test_set, test_labels) = readData('/Test/*/*.txt', 10)

print('size', test_set.size, test_labels.size)
# training_set = data[:training_set_size]
# training_labels = labels[:training_set_size]
# test_set = data[training_set_size:]
# test_labels = labels[training_set_size:]

model = keras.Sequential([
    keras.layers.Conv1D(100, 250, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])

model.fit(training_set, training_labels, epochs=n_epochs, callbacks=[csv_logger])
# model.fit(training_set, training_labels, epochs=n_epochs)

_, test_loss = model.evaluate(test_set, test_labels)
model.save("model_{}_{}.h5".format(n_samples,n_epochs))
print('Test loss', test_loss)
# model.save(path + '/tmp/e' + str(n_epochs) + 'n' + str(n_samples) + 'model.h5')

