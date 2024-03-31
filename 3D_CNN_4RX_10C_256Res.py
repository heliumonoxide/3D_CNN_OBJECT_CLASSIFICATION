import os, sys
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
#from imutils import paths
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def model_bryan():
    ##merancang model CNN##
    model1 = Sequential()
    # layer 1
    model1.add(Conv2D(8, (3, 3), padding='same', input_shape=(256, 256,
                                                              2)))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
    model1.add(Activation('relu'))  # menghilangkan nilai negatif
    model1.add(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9
    # layer 2
    model1.add(Conv2D(16, (3, 3), padding='same'))
    model1.add(Activation('relu'))
    model1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # layer Hidden
    model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
    model1.add(Dense(10, input_dim=1, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    # layer Klasifikasi
    model1.add(Dense(2, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

    # melihat desain model CNN yang sudah dibuat
    model1.summary()
    return model1

def plot_acc_lost(MODEL_SAVE_FOLDER_PATH, history):
    plt.figure(figsize=(16, 12))
    plt.plot(history.history['acc'], 'r--', label='Accuracy of Training Data')
    plt.plot(history.history['val_acc'], 'b--', label='Accuracy of Validation Data')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.ylim(0, 1)
    plt.legend()

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + 'plot_acc/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH + 'plot_acc/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + 'plot_acc/' + 'acc_plot.jpg')
    # plt.show()
    plt.close()

    plt.figure(figsize=(16, 12))
    plt.plot(history.history['loss'], 'g-.', label='Loss of Training Data')
    plt.plot(history.history['val_loss'], 'c-.', label='Loss of Validation Data')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0, 1)
    plt.legend()

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + 'plot_loss/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH + 'plot_loss/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + 'plot_loss/' + 'loss_plot.jpg')
    #plt.show()
    plt.close()

    plt.figure(figsize=(16, 12))
    plt.plot(history.history['acc'], 'r--', label='Accuracy of Training Data')
    plt.plot(history.history['val_acc'], 'b--', label='Accuracy of Validation Data')
    plt.plot(history.history['loss'], 'g-.', label='Loss of Training Data')
    plt.plot(history.history['val_loss'], 'c-.', label='Loss of Validation Data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')

    plt.legend()

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + 'plot_acc_lost/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH + 'plot_acc_lost/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + 'plot_acc_lost/' + 'acc_loss_plot.jpg')
    # plt.show()
    plt.close()

def model_train(MODEL_SAVE_FOLDER_PATH, train_data, train_label, valid_data, valid_label, Batch_size, Epochs, n_fold, alpha, model, object):

    Train_data = train_data
    Train_label = train_label
    Valid_data = valid_data
    Valid_label = valid_label
    lb = LabelBinarizer()
    Train_label=lb.fit_transform(Train_label)
    Valid_label=lb.fit_transform(Valid_label)

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/')

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/'+object+'/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/'+object+'/')

    model_path = MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/'+object+'/'+'best_weight.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)


    # model=model_bryan()
    # setting optimizer
    opt = Adam(
        learning_rate=alpha)  # bikin variabel namanya opt, dengan learning rate biasanya antara 0.0001 - 0.1. jika terlalu besar akan terjadi overshooting
    model.compile(loss='binary_crossentropy', optimizer=opt,
                   metrics=['acc'])  # untuk 2 kelas, gunakan binary_crossentropy, bila lebih categorical_crossentropy


    # training model cnn
    start = time.time()
    history = model.fit(Train_data, Train_label, callbacks=[checkpoint], validation_data=(Valid_data, valid_label), epochs=Epochs,
                   batch_size=Batch_size)  # X= data, Y = label, epoch=jumlah pelatihan. batch_size= jumlah citra yang digunakan dalam 1 kali pembacaan
    end = time.time()

    # mencatat computational cost dalam bentuk .txt
    orig_stdout = sys.stdout
    nama_txt='computational_cost/'+object+'_ComputationalCost_CNN2Rx_'+str(n_fold)+'.txt'
    f = open(nama_txt, 'w')
    sys.stdout = f
    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    sys.stdout = orig_stdout
    f.close()

    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    plot_acc_lost(MODEL_SAVE_FOLDER_PATH+str(n_fold)+'/', history)
    print("\n training is done!! \n")

    # Print Validation Loss and Accuracy
    test_eval = model.evaluate(Valid_data, Valid_label)
    print('Test Loss: ', test_eval[0])
    print('Test Accuracy: ', test_eval[1])

    return history

def k_fold_training(n_folds,model_save_path,batch_size,epochs):
    model_history=[]
    alpha = 0.0001
    objek = "Human_Bunny"
    model = model_bryan()
    for i in range(n_folds):
        print("Training on Fold: ", i+1)
        # data_latih, data_uji, label_data_latih, label_data_uji = create_data_train_test(kelas_1_rx1,kelas_1_rx2,kelas_2_rx1,kelas_2_rx2)
        data_latih = np.load("temp_train_test_data/trainX_256_dummy.npy")
        data_uji = np.load("temp_train_test_data/testX_256_dummy.npy")
        label_data_latih = np.load("temp_train_test_data/trainY_256_dummy.npy")
        label_data_uji = np.load("temp_train_test_data/testY_256_dummy.npy")
        history=model_train(model_save_path, data_latih, label_data_latih, data_uji, label_data_uji, batch_size,epochs, i+1, alpha, model, objek)
        model_history.append(history)
        # alpha = alpha * 0.1
    return model_history

# data_rx1='processed_data/Data_Human_Rx1_1m.csv'
# data_rx2='processed_data/Data_Human_Rx2_1m.csv'
# data_rx3='processed_data/Data_WalkingHuman_Rx1_1m.csv'
# data_rx4='processed_data/Data_WalkingHuman_Rx2_1m.csv'
jumlah_n_folds=4
folder_simpan_weight='processed_cnn/weight_hasil/'
Batch_Size = 32
Epochs=30
k_fold_training(jumlah_n_folds,folder_simpan_weight,Batch_Size,Epochs)