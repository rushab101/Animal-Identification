import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D
from tensorflow.python.keras.datasets  import cifar10
from tensorflow.python.keras.callbacks import TensorBoard
import time
import pickle

NAME = "Cats-vs-dogs-vs-monkeys-squirrel-CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
gpu_option=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)

pickle_in = open("X.pickle_new","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle_new","rb")
y = pickle.load(pickle_in)

X = X/255.0


dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]


for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      validation_split=0.3,
                      )

model.save('64x3-new_modelv2.model')