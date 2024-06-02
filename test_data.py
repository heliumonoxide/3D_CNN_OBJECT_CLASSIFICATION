# import os, sys
import numpy as np
# from sklearn.preprocessing import LabelBinarizer # type: ignore
# import tensorflow as tf
# from tensorflow.keras.models import Sequential # type: ignore
# from tensorflow.keras.layers import Conv3D # type: ignore
# from tensorflow.keras.layers import MaxPooling3D # type: ignore
# # from tensorflow.keras.layers import Activation # type: ignore
# from tensorflow.keras.layers import AveragePooling3D # type: ignore
# from tensorflow.keras.layers import Flatten # type: ignore
# from tensorflow.keras.layers import Dense # type: ignore
# from tensorflow.keras.optimizers import Adam # type: ignore
# import matplotlib.pyplot as plt # type: ignore
# #from imutils import paths
# from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
# import time

# def newmodel2():
#     ##merancang model CNN##
#     model1 = Sequential()

#     # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid", input_shape=(256, 256, 256, 3)))
#     # layer 1
#     model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu', input_shape=(256, 256, 256, 3)))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
#     model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya

#     model1.add(
#         MaxPooling3D(pool_size=(9, 9, 9), strides=(9, 9, 9)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9

#     # layer 2
#     model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(MaxPooling3D(pool_size=(9, 9, 9), strides=(9, 9, 9)))

#     # layer 3
#     model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))

#     # layer Hidden
#     model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
#     model1.add(Dense(64, input_dim=1, activation='relu'))
#     model1.add(Dense(32, activation='relu'))
#     model1.add(Dense(24, activation='relu'))
#     model1.add(Dense(16, activation='relu'))
#     model1.add(Dense(12, activation='relu'))
#     model1.add(Dense(8, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # layer Klasifikasi
#     model1.add(Dense(4, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

#     # melihat desain model CNN yang sudah dibuat
#     model1.summary()
#     return model1

# def newmodel():
#     ##merancang model CNN##
#     model1 = Sequential()

#     # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid", input_shape=(256, 256, 256, 3)))
#     # layer 1
#     model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu', input_shape=(256, 256, 256, 3)))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
#     model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya

#     model1.add(
#         MaxPooling3D(pool_size=(9, 9, 9), strides = (9, 9, 9)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9
    
#     # layer 2
#     model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
#     model1.add(MaxPooling3D(pool_size=(9, 9, 9), strides = (9, 9, 9)))

#     # layer 3
#     # model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
#     # model1.add(Conv3D(32, (3, 3, 3), padding='same', activation='relu'))
#     # model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))

#     # layer Hidden
#     model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
#     model1.add(Dense(864, input_dim=1, activation='relu'))
#     model1.add(Dense(648, activation='relu'))
#     model1.add(Dense(436, activation='relu'))
#     model1.add(Dense(220, activation='relu'))
#     model1.add(Dense(100, activation='relu'))
#     model1.add(Dense(50, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # model1.add(Dense(10, activation='relu'))
#     # layer Klasifikasi
#     model1.add(Dense(4, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

#     # melihat desain model CNN yang sudah dibuat
#     model1.summary()
#     return model1

# ==========================================================================
# ===================== PLOTTING FOR DATA ILLUSTRATION =====================
# ==========================================================================

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Given data
class1_data_rx1='processed_data/datafft_Anoa_1_1mtr.npy'
class1_data_rx2='processed_data/datafft_Makaka_Nemestrina_1_1mtr.npy'
class1_data_rx3='processed_data/datafft_OrangUtan_1_1mtr.npy'
    
# a = np.load(class1_data_rx1)
# b = np.load(class1_data_rx2)
# c = np.load(class1_data_rx3)

# a = np.float32(a)
# b = np.float32(b)
# c = np.float32(c)

# a = np.round(a, decimals=5)
# b = np.round(b, decimals=5)
# c = np.round(c, decimals=5)

# print(a.dtype)
# print(type(a[0]))

# aa = []
# bb = []
# cc = []

# for k in range(32):
#     aa.append(a[k])
#     bb.append(b[k])
#     cc.append(c[k])
#     print("appending: ")
#     print(k)

# print(aa)
# print(bb)
# print(cc)

# del a
# del b
# del c

# # Get absolute values
# abs_a = [abs(x) for x in aa]
# print("abs_a done")
# abs_b = [abs(x) for x in bb]
# print("abs_b done")
# abs_c = [abs(x) for x in cc]
# print("abs_c done")

# # Normalize RGB values
# a_norm = [val / max(abs_a) for val in abs_a]
# print("Norm a done")
# b_norm = [val / max(abs_b) for val in abs_b]
# print("Norm b done")
# c_norm = [val / max(abs_c) for val in abs_c]
# print("Norm c done")

# del abs_a
# del abs_b
# del abs_c

# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot
# for ix in range(len(a_norm)):
#     print("Scattering: ")
#     print(ix)
#     for iy in range(len(b_norm)):
#         for iz in range(len(c_norm)):
#             ax.scatter(ix, iy, iz, c=[(a_norm[ix], b_norm[iy], c_norm[iz])], marker='o') 

# # Set labels
# ax.set_xlabel('X (Rx1)')
# ax.set_ylabel('Y (Rx2)')
# ax.set_zlabel('Z (Rx3)')

# plt.title("Data Representasion of 3D Tensor Data")
# plt.savefig('./processed_cnn/figures/DataRepresentation_CNN3Rx.jpg')
# plt.savefig('./processed_cnn/figures/DataRepresentation_CNN3Rx.pdf')

# ax = fig.add_subplot(111, projection='3d')
# for ix in range(len(a_norm)):
#     print("Scattering: ")
#     print(ix)
#     for iy in range(len(b_norm)):
#         for iz in range(len(c_norm)):
#             ax.scatter(ix, iy, iz, c=None, marker='o') 

# ax.set_xlabel('X (Rx1)')
# ax.set_ylabel('Y (Rx2)')
# ax.set_zlabel('Z (Rx3)')

# plt.title("Data Representasion of 3D Tensor Data (Uncolorized)")
# plt.savefig('./processed_cnn/figures/DataRepresentation_CNN3Rx_Uncolorized.jpg')
# plt.savefig('./processed_cnn/figures/DataRepresentation_CNN3Rx_Uncolorized.pdf')

# plt.show()
# plt.close()

# ==========================================================================
# =============== PLOTTING FOR DATA ILLUSTRATION 1 Receiver ================
# ==========================================================================
import math

a = np.load(class1_data_rx1)
b = np.load(class1_data_rx2)
c = np.load(class1_data_rx3)

a = np.float32(a)
b = np.float32(b)
c = np.float32(c)

original_array = list(range(1, len(a)+1))
min_orig = 1
max_orig = len(a)
min_mapped = 1.0
max_mapped = 10.0

mapped_array = [min_mapped + ((x - min_orig) / (max_orig - min_orig)) * (max_mapped - min_mapped) for x in original_array]

# For verification, let's print the first and last 10 elements
first_10_elements = mapped_array[:10]
last_10_elements = mapped_array[-10:]


# plt.plot(a, label='Rx1 Anoa')
plt.plot(mapped_array, b, label='Rx1 Makaka Nemestrina')
# plt.plot(c, label='Rx1 Orangutan')
# plt.axis((0, 50, 0, 1))
plt.ylabel('Magnitude')
plt.xlabel('Frequency (Hz)')
plt.legend()
plt.show()
plt.close()
print(len(a))

# import matplotlib.pyplot as plt

# L = len(a)
# Fs = 1000 # Frequency sampling
# T = 1/Fs        
# t = np.arange(0, L) * T 
# sp = np.fft.fft(np.sin(t))
# freq = np.fft.fftfreq(t.shape[-1])
# plt.plot(freq, sp.real, freq, sp.imag)
# # plt.legend()

# plt.show()
