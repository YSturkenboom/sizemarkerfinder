import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import random
import logging
from tqdm import tqdm

def make_labels(f):
    data = None
    with open(f,'r') as file:
        #print(np.array(list(csv.reader(file))))
        data = np.array(list(csv.reader(file)))[1:]
    #data = np.transpose(data)
    sizemarker_pos = []
    # print(data)
    for experiment in data:
        # print(experiment)
        # [print(datapoint) for datapoint in experiment]
        # [int(datapoint[1]) for datapoint in experiment]
        # [int(datapoint[1]) for datapoint in experiment if datapoint[4] != '-1']
        
        if (experiment[4] != '-1'):
            sizemarker_pos.append(int(experiment[4]))
        
        """
        else:
         
            sizemarker_pos = list(int(datapoint[1]) for datapoint in np.transpose(experiment) if datapoint[4] != '-1')
            
            # experimental: if missing sizemarkers add -1's
            for i in range(31- len(sizemarker_pos)):
                sizemarker_pos.append(-1)

            print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))

            labels = np.vstack((labels, sizemarker_pos))
        """
    
    # print("Sizemarker positions", sizemarker_pos, "Sizemarkers in experiment", len(sizemarker_pos))
    
    # experimental: if missing sizemarkers add -1's
    for i in range(31- len(sizemarker_pos)):
        sizemarker_pos.append(-1)
        
    labels = np.array(sizemarker_pos)
    
    # print(labels.shape)
    # print(data.shape)

    # select columns of interest: RFU and time
    data = data[:,1:3]
    data = np.expand_dims(data, axis=0)
    labels = np.expand_dims(labels, axis=0)
    
    return [data, labels]
        
if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG, filename="log.txt")
    logging.info("Test Accuracy, Test Loss")

    # Do not use GPU (use CPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #config = tf.ConfigProto()
    #tf.keras.backend.set_session(tf.Session(config=config))

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

    #data = np.array([])

    # Create model
    model = keras.Sequential([
        keras.layers.Conv1D(100, 31, activation='linear', input_shape=(25000, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(31, activation='linear')
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    saved_model = None
    # Train model per file?              
    files = [f for f in glob.glob("./Data/*.txt")]
    train_set_size = int(round(n_samples * 0.8))
    random.shuffle(files)
    files = files[:n_samples]
    train_set = files[:train_set_size]
    test_set = files[train_set_size:]
    
    # Build test set
    test_labels = np.expand_dims(np.array([]), axis=0)
    test_samples = np.array([])
    for test_file in test_set:
        [data, labels] = make_labels(test_file)
        if test_samples.size == 0:
            test_samples = data
        else:
            # print(data.shape)
            # print(test_samples.shape)
            test_samples = np.vstack((test_samples,  data))
        if test_labels.size == 0:
            test_labels = labels
        else:
            # print(labels.shape)
            # print(test_labels.shape)
            test_labels = np.vstack((test_labels, labels))
        #print(test_labels.shape)   
        #print(test_samples.shape)

    # test_labels = np.expand_dims(np.array(test_labels), axis=0)
    # test_samples = np.expand_dims(np.array(test_samples), axis=0)
    for _ in range(n_epochs):
        for f in tqdm(train_set):
            """
            with open(f, 'r') as file:
                if (data.size == 0):
                    data = np.array(list(csv.reader(file))[1:])
                else:    
                    data = np.dstack((data, list(csv.reader(file))[1:]))
            """
            [data, labels] = make_labels(f)

            # 80/20 train/test split
            """
            training_set_size = int(round(n_samples * 0.8))

            training_set = data[:training_set_size]
            training_labels = labels[:training_set_size]
            test_set = data[training_set_size:]
            test_labels = labels[training_set_size:]
            print(training_set[0].shape)



            model.fit(training_set, training_labels, epochs=n_epochs)
            """
            if saved_model:
                model = keras.models.load_model('modeldraft1.h5')
                # print("loading saved model")
                # saved_model = saved_model.fit(data, labels, epochs=1)
            # else:
                # print(data.shape)
                # saved_model = model.fit(data, labels, epochs=1)
                
            model.fit(data, labels, epochs=1)
            del data
            saved_model = True
            model.save('modeldraft1.h5')
            K.clear_session()
                
        model = keras.models.load_model('modeldraft1.h5')
        # test_samples = test_samples[0,:,:,:]
        test_loss, test_acc = model.evaluate(test_samples, test_labels)
        print('Test accuracy:', test_acc, 'Test loss', test_loss)
        logging.info('%s, %s', test_acc, test_loss)