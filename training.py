import sys
import os

# Do not use GPU (use CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

n_samples = 100
n_epochs = 100

# read cmd line args
if (len(sys.argv) > 1):
	n_samples = int(sys.argv[1])
	if (n_samples > 1000):
		print("sample size exceeds available samples")
		sys.exit()
if (len(sys.argv) > 2):
	n_epochs  = int(sys.argv[2])

print('n_samples', n_samples, 'n_epochs', n_epochs)

data = np.array([])

files = [f for f in glob.glob("./smalltest/*.txt")]
for f in files:
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

        print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

        labels = sizemarker_pos
    else:
        sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')
        
        # experimental: if missing sizemarkers add -1's
        for i in range(31- len(sizemarker_pos)):
            sizemarker_pos.append(-1)

        print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

        labels = np.vstack((labels, sizemarker_pos))

print(labels.shape)
print(data.shape)

# select columns of interest: RFU and time
data = np.transpose(data[:,1:3], (0, 2, 1))
print(data.shape)

print(data[0])
print(labels[0])

# 80/20 train/test split
training_set_size = int(round(n_samples * 0.8))

training_set = data[:training_set_size]
training_labels = labels[:training_set_size]
test_set = data[training_set_size:]
test_labels = labels[training_set_size:]

print(training_set[0].shape)

model = keras.Sequential([
    keras.layers.Conv1D(100, 31, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(31, activation='linear')
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(training_set, training_labels, epochs=n_epochs)
test_loss, test_acc = model.evaluate(test_set, test_labels)

print('Test accuracy:', test_acc, 'Test loss', test_loss)

model.save('modeldraft1.h5')
