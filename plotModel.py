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

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(25000, 2)),
    keras.layers.Dense(10000, activation='linear'),
    keras.layers.Dense(5000, activation='linear'),
    keras.layers.Dense(500, activation='linear'),
    keras.layers.Dense(100, activation='linear'),
    keras.layers.Dense(31, activation='linear'),
])
              
from tensorflow.keras.utils import plot_model
plot_model(model, show_shapes=True, to_file=path + "/model.png")