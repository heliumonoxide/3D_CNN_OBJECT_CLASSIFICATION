import os, sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv3D # type: ignore
from tensorflow.keras.layers import MaxPooling3D # type: ignore
# from tensorflow.keras.layers import Activation # type: ignore
from tensorflow.keras.layers import AveragePooling3D # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt # type: ignore
#from imutils import paths
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import time

def newmodel2():
    ##merancang model CNN##
    model1 = Sequential()

    # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid", input_shape=(256, 256, 256, 3)))
    # layer 1
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu', input_shape=(256, 256, 256, 3)))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya

    model1.add(
        MaxPooling3D(pool_size=(9, 9, 9), strides=(9, 9, 9)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9

    # layer 2
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(MaxPooling3D(pool_size=(9, 9, 9), strides=(9, 9, 9)))

    # layer 3
    model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
    model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
    model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))

    # layer Hidden
    model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
    model1.add(Dense(64, input_dim=1, activation='relu'))
    model1.add(Dense(32, activation='relu'))
    model1.add(Dense(24, activation='relu'))
    model1.add(Dense(16, activation='relu'))
    model1.add(Dense(12, activation='relu'))
    model1.add(Dense(8, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # layer Klasifikasi
    model1.add(Dense(4, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

    # melihat desain model CNN yang sudah dibuat
    model1.summary()
    return model1

def newmodel():
    ##merancang model CNN##
    model1 = Sequential()

    # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid", input_shape=(256, 256, 256, 3)))
    # layer 1
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu', input_shape=(256, 256, 256, 3)))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya

    model1.add(
        MaxPooling3D(pool_size=(9, 9, 9), strides = (9, 9, 9)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9
    
    # layer 2
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(MaxPooling3D(pool_size=(9, 9, 9), strides = (9, 9, 9)))

    # layer 3
    # model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
    # model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
    # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))

    # layer Hidden
    model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
    model1.add(Dense(864, input_dim=1, activation='relu'))
    model1.add(Dense(648, activation='relu'))
    model1.add(Dense(436, activation='relu'))
    model1.add(Dense(220, activation='relu'))
    model1.add(Dense(100, activation='relu'))
    model1.add(Dense(50, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # layer Klasifikasi
    model1.add(Dense(4, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

    # melihat desain model CNN yang sudah dibuat
    model1.summary()
    return model1