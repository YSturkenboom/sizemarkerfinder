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
experimentName = 'Sep25-H3-NoHarmonica-SGD-YNormalized'

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

n_samples = 800
n_samples_val = 200
n_epochs = 1000
model_version = 'v1'
shuffle = True
normalize = True
generate_log = True
checkpoints = False
save_model = True
create_plots_at_epochs = [0,1,2,3,4,5,10,25,50,100,200,300,400,500,600,700,800,900,999]
hyperparams = [n_samples, n_samples_val, n_epochs, shuffle, normalize, model_version, create_plots_at_epochs]
hyperparam_names = ['n_samples', 'n_samples_val', 'n_epochs', 'shuffle', 'normalize', 'model_version', 'create_plots_at_epochs']

files = []
filesNoDrop = []
filesNoHarm = []
filesVal = []
filesValNoDrop = []
filesValNoHarm = []
filesTest = []

def generateProfile(data, predictions, plt, n1, n2, n3, title):
  profile = data[0]
  pred_array = np.zeros(25000)
  for pred in predictions[0]:
    pred = pred * 25000.0
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
    print(predictions)

    generateProfile(pred_data, predictions, plt, n1, n2, n3, title)

def generateProfiles(current_model, epoch):
  plt.figure(figsize=(30,15))
  plt.suptitle("Experiment " + experimentName + ": Training set profiles at epoch " + str(epoch), fontsize=16)
  # makePredictions(current_model, files[0], plt, 3, 5, 1, 'Sample from normal training set')
  # makePredictions(current_model, files[1], plt, 3, 5, 2, 'Sample from normal training set')
  # makePredictions(current_model, files[2], plt, 3, 5, 3, 'Sample from normal training set')
  # makePredictions(current_model, files[3], plt, 3, 5, 4, 'Sample from normal training set')
  # makePredictions(current_model, files[4], plt, 3, 5, 5, 'Sample from normal training set')
  # makePredictions(current_model, filesNoDrop[0], plt, 3, 5, 6, 'Sample from no-drop training set')
  # makePredictions(current_model, filesNoDrop[1], plt, 3, 5, 7, 'Sample from no-drop training set')
  # makePredictions(current_model, filesNoDrop[2], plt, 3, 5, 8, 'Sample from no-drop training set')
  # makePredictions(current_model, filesNoDrop[3], plt, 3, 5, 9, 'Sample from no-drop training set')
  # makePredictions(current_model, filesNoDrop[4], plt, 3, 5, 10, 'Sample from no-drop training set')
  makePredictions(current_model, filesNoHarm[0], plt, 3, 5, 11, 'Sample from no-harmonica training set')
  makePredictions(current_model, filesNoHarm[1], plt, 3, 5, 12, 'Sample from no-harmonica training set')
  makePredictions(current_model, filesNoHarm[2], plt, 3, 5, 13, 'Sample from no-harmonica training set')
  makePredictions(current_model, filesNoHarm[3], plt, 3, 5, 14, 'Sample from no-harmonica training set')
  makePredictions(current_model, filesNoHarm[4], plt, 3, 5, 15, 'Sample from no-harmonica training set')
  plt.savefig(path+'/experiments/'+experimentName+'/plot-ep'+str(epoch)+'-train.png')
  plt.close()

  plt.figure(figsize=(30,15))
  plt.suptitle("Experiment " + experimentName + ": Validation set profiles at epoch " + str(epoch), fontsize=16)
  # makePredictions(current_model, filesVal[0] , plt, 3,5,1, 'Sample from normal validation set')
  # makePredictions(current_model, filesVal[1] , plt, 3,5,2, 'Sample from normal validation set')
  # makePredictions(current_model, filesVal[2] , plt, 3,5,3, 'Sample from normal validation set')
  # makePredictions(current_model, filesVal[3] , plt, 3,5,4, 'Sample from normal validation set')
  # makePredictions(current_model, filesVal[4] , plt, 3,5,5, 'Sample from normal validation set')
  # makePredictions(current_model, filesValNoDrop[0] , plt, 3,5,6, 'Sample from no-drop validation set')
  # makePredictions(current_model, filesValNoDrop[1] , plt, 3,5,7, 'Sample from no-drop validation set')
  # makePredictions(current_model, filesValNoDrop[2] , plt, 3,5,8, 'Sample from no-drop validation set')
  # makePredictions(current_model, filesValNoDrop[3] , plt, 3,5,9, 'Sample from no-drop validation set')
  # makePredictions(current_model, filesValNoDrop[4] , plt, 3,5,10, 'Sample from no-drop validation set')
  makePredictions(current_model, filesValNoHarm[0] , plt, 3,5,11, 'Sample from no-harmonica validation set')
  makePredictions(current_model, filesValNoHarm[1] , plt, 3,5,12, 'Sample from no-harmonica validation set')
  makePredictions(current_model, filesValNoHarm[2] , plt, 3,5,13, 'Sample from no-harmonica validation set')
  makePredictions(current_model, filesValNoHarm[3] , plt, 3,5,14, 'Sample from no-harmonica validation set')
  makePredictions(current_model, filesValNoHarm[4] , plt, 3,5,15, 'Sample from no-harmonica validation set')
  plt.savefig(path+'/experiments/'+experimentName+'/plot-ep'+str(epoch)+'-val.png')
  plt.close()

  plt.figure(figsize=(30,15))
  plt.suptitle("Experiment " + experimentName + ": Test set profiles at epoch " + str(epoch), fontsize=16)
  makePredictions(current_model, filesTest[0], plt, 1,5,1, 'Sample from normal test set')
  makePredictions(current_model, filesTest[1], plt, 1,5,2, 'Sample from normal test set')
  makePredictions(current_model, filesTest[2], plt, 1,5,3, 'Sample from normal test set')
  makePredictions(current_model, filesTest[3], plt, 1,5,4, 'Sample from normal test set')
  makePredictions(current_model, filesTest[4], plt, 1,5,5, 'Sample from normal test set')
  makePredictions(current_model, filesTest[5], plt, 3,5,6, 'Sample from no-drop test set')
  makePredictions(current_model, filesTest[6], plt, 3,5,7, 'Sample from no-drop test set')
  makePredictions(current_model, filesTest[7], plt, 3,5,8, 'Sample from no-drop test set')
  makePredictions(current_model, filesTest[8], plt, 3,5,9, 'Sample from no-drop test set')
  makePredictions(current_model, filesTest[9], plt, 3,5,10, 'Sample from no-drop test set')
  makePredictions(current_model, filesTest[10], plt, 3,5,11, 'Sample from no-harmonica test set')
  makePredictions(current_model, filesTest[11], plt, 3,5,12, 'Sample from no-harmonica test set')
  makePredictions(current_model, filesTest[12], plt, 3,5,13, 'Sample from no-harmonica test set')
  makePredictions(current_model, filesTest[13], plt, 3,5,14, 'Sample from no-harmonica test set')
  makePredictions(current_model, filesTest[14], plt, 3,5,15, 'Sample from no-harmonica test set')
  plt.savefig(path+'/experiments/'+experimentName+'/plot-ep'+str(epoch)+'-test.png')
  plt.close()

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
    fileList = [f for f in glob.glob(path + data_path)]
    # random.shuffle(files)
    data = np.zeros((amount, 25000, 3))
    labels = np.zeros((amount, 31))
    # print(data_path)
    for idx, f in tqdm(enumerate(fileList[:amount])):
        with open(f, 'r') as file:
            # stuff = np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            # print('stuff', len(stuff), stuff.size, stuff)
            print(f)
            loadeddata =  np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1,2,4))
            data[idx] = loadeddata
            
            # print('loaded', loadeddata)

            sizemarker_pos = list(int(datapoint[0]) for datapoint in loadeddata if datapoint[2] != -1)
            for _ in range(31- len(sizemarker_pos)):
              sizemarker_pos.append(-1)

            # print(sizemarker_pos)

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

    return (data[:,:,:2], labels, fileList[:amount])

# (data, labels, files) = readData('/Data/*.txt', n_samples)
# (dataNoDrop, labelsNoDrop, filesNoDrop) = readData('/DataNoDrop/*.txt', n_samples)
(dataNoHarm, labelsNoHarm, filesNoHarm) = readData('/DataNoHarm2/*.txt', n_samples*3)
training_set = dataNoHarm
training_labels = labelsNoHarm

# (valData, valLabels, filesVal) = readData('/Validation/Data/*.txt', n_samples_val)
# (valDataNoDrop, valLabelsNoDrop, filesValNoDrop) = readData('/Validation/DataNoDrop/*.txt', n_samples_val)
(valDataNoHarm, valLabelsNoHarm, filesValNoHarm) = readData('/Validation/DataNoHarm2/*.txt', n_samples_val)
val_set = valDataNoHarm
val_labels = valLabelsNoHarm

(test_set, test_labels, filesTest) = readData('/Test/*/*.txt', n_samples_val)

if (normalize):
  # normalize: divide RFU by 1000, time by 25000, labels by 1500
  training_set = np.divide(training_set, 25000)
  training_labels = np.divide(training_labels, 25000)
  test_set = np.divide(test_set, 25000)
  test_labels = np.divide(test_labels, 25000)
  val_set = np.divide(val_set, 25000)
  val_labels = np.divide(val_labels, 25000)
# select columns of interest: RFU and time
# test_set = np.transpose(test_set[:,1:3], (0, 2, 1))

model = keras.Sequential([
    keras.layers.Conv1D(31, 250, activation='linear', input_shape=(25000, 2)),
    keras.layers.Flatten(),
#     keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear')
])

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(25000, 2)),
#     keras.layers.Dense(5000, activation='linear'),
#     keras.layers.Dense(1000, activation='linear'),
#     keras.layers.Dense(500, activation='linear'),
#     keras.layers.Dense(100, activation='linear'),
#     keras.layers.Dense(31, activation='linear'),
# ])
# optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0, amsgrad=False)
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, clipnorm=1.0)

model.compile(optimizer=optimizer,
              loss='mean_squared_error')
              
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, to_file=path + "/experiments/" + experimentName +  "/model.png")

callbacks = []
if (checkpoints):
  callbacks.append(checkpointer)
if (generate_log):
  callbacks.append(csv_logger)
if (len(create_plots_at_epochs) > 0):
  plot_callback = PlotCallback()
  callbacks.append(plot_callback)
  
startFit = time.time()
model.fit(training_set, training_labels, validation_data=(val_set, val_labels), shuffle=shuffle, batch_size=32, epochs=n_epochs, callbacks=callbacks)

test_loss = model.evaluate(test_set, test_labels)
with open(path + '/experiments/' + experimentName + '/hyperparams.txt', 'a') as f:
  f.write('model_fit_time: ' + str(time.time() - startFit) + '\n')
  f.write('test_loss: ' + str(test_loss))
print('Test loss', test_loss)


print('Test loss', test_loss)

if (save_model):
  model.save(path + "/latest_model.h5")
  # model.save_weights(path + '/experiments/' + experimentName + '/weights.h5')
