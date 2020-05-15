from data_preprocess import FiducialDataProcess
import ipdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import optimizers

np.random.seed(0)

x_train = np.genfromtxt('pca_data/train_pca_compound.csv', delimiter=',')
np.random.shuffle(x_train)
x_test =  np.genfromtxt('pca_data/test_pca_compound.csv', delimiter=',')
np.random.shuffle(x_test)

x_train_pca = x_train[:,2:]#/x_train[:,2:].max(axis=0)
y_train_emot1 = x_train[:,1].astype(int)
y_train_emot1 -= 8*np.ones(len(y_train_emot1), dtype='int64')

x_test_pca = x_test[:,2:]#/np.linalg.norm(x_test[:,2:])#/x_test[:,2:].max(axis=0)
y_test_emot1 = x_test[:,1].astype(int)
y_test_emot1 -= 8*np.ones(len(y_test_emot1), dtype='int64')


model = keras.Sequential([
	keras.layers.Dense(64, activation = tf.nn.leaky_relu),
	keras.layers.Dropout(0.5),
	keras.layers.Dense(48, activation = tf.nn.leaky_relu),
	keras.layers.Dropout(0.2),
    keras.layers.Dense(15, activation = tf.nn.softmax)
])

model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(x_train_pca, y_train_emot1, validation_data = (x_test_pca,y_test_emot1), batch_size = 12,epochs = 50)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the number of epochs
epochs = range(len(acc))

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, color='blue', label='Train')
plt.plot(epochs, val_acc, color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

_ = plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, color='blue', label='Train')
plt.plot(epochs, val_loss, color='orange', label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()