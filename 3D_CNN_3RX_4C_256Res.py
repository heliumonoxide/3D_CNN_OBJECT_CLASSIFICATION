import os, sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv3D # type: ignore
from tensorflow.keras.layers import MaxPooling3D # type: ignore
# from tensorflow.keras.layers import Activation # type: ignore
from tensorflow.keras.layers import AveragePooling3D # type: ignore
from tensorflow.keras.layers import Flatten # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
#from imutils import paths
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

# Documentation of 3D Convolution: https://keras.io/api/layers/convolution_layers/convolution3d/ and https://www.analyticsvidhya.com/blog/2022/05/building-a-3d-cnn-in-tensorflow/ 

def model_bryan():
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

# Define a data generator function
def data_generator(data, labels, batch_size):
    num_samples = len(data)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield data[excerpt], labels[excerpt]

def plot_acc_lost(MODEL_SAVE_FOLDER_PATH, history, n_fold):
    plt.figure(figsize=(16, 12))
    plt.plot(history.history['acc'], 'r--', label='Accuracy of Training Data')
    plt.plot(history.history['val_acc'], 'b--', label='Accuracy of Validation Data')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.ylim(0, 1)
    plt.legend()

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_acc/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH+ str(n_fold) + '/' + 'plot_acc/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_acc/' + 'acc_plot.jpg')
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

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_loss/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_loss/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_loss/' + 'loss_plot.jpg')
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

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_acc_lost/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_acc_lost/')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + str(n_fold) + '/' + 'plot_acc_lost/' + 'acc_loss_plot.jpg')
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

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH+object+'/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH+object+'/')

    if not os.path.exists(MODEL_SAVE_FOLDER_PATH+object+'/'+str(n_fold)+'/'):
        os.mkdir(MODEL_SAVE_FOLDER_PATH+object+'/'+str(n_fold)+'/')

    model_path = MODEL_SAVE_FOLDER_PATH+object+'/'+str(n_fold)+'/'+'best_weight.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)


    # model=model_bryan()
    # setting optimizer
    opt = Adam(
        learning_rate=alpha)  # bikin variabel namanya opt, dengan learning rate biasanya antara 0.0001 - 0.1. jika terlalu besar akan terjadi overshooting
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                   metrics=['acc'])  # untuk 2 kelas, gunakan binary_crossentropy, bila lebih categorical_crossentropy

    train_generator = data_generator(Train_data, Train_label, Batch_size)
    valid_generator = data_generator(Valid_data, Valid_label, Batch_size)

    # training model cnn
    start = time.time()
    # history = model.fit(Train_data, Train_label, callbacks=[checkpoint], validation_data=(Valid_data, valid_label), epochs=Epochs,
    #                batch_size=Batch_size)  # X= data, Y = label, epoch=jumlah pelatihan. batch_size= jumlah citra yang digunakan dalam 1 kali pembacaan
    history = model.fit_generator(train_generator, 
                                  steps_per_epoch=len(Train_data)//Batch_size, 
                                  epochs=Epochs, 
                                  validation_data=valid_generator, 
                                  validation_steps=len(Valid_data)//Batch_size, 
                                  callbacks=[checkpoint])
    end = time.time()

    # mencatat computational cost dalam bentuk .txt
    orig_stdout = sys.stdout
    nama_txt='processed_cnn/computational_cost/'+object+'_ComputationalCost_CNN3Rx_'+str(n_fold)+'.txt'
    f = open(nama_txt, 'w')
    sys.stdout = f
    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    sys.stdout = orig_stdout
    f.close()

    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    plot_acc_lost(MODEL_SAVE_FOLDER_PATH+object+'/', history)
    print("\n training is done!! \n")

    # # Print Validation Loss and Accuracy
    # test_eval = model.evaluate(Valid_data, Valid_label)
    # print('Test Loss: ', test_eval[0])
    # print('Test Accuracy: ', test_eval[1])

    return history

def k_fold_training(n_folds,model_save_path,batch_size,epochs):
    model_history=[]
    alpha = 0.0001
    objek = "OU-MN-MT-Anoa-Test-stride9"
    model = model_bryan()
    data_latih = np.load("temp_train_test_data/trainX_256_4Class_OU-MN-MT-Anoa_25data.npy")
    label_data_latih = np.load("temp_train_test_data/trainY_256_4Class_OU-MN-MT-Anoa_25data.npy")
    data_uji = np.load("temp_train_test_data/testX_256_4Class_OU-MN-MT-Anoa_25data.npy")
    label_data_uji = np.load("temp_train_test_data/testY_256_4Class_OU-MN-MT-Anoa_25data.npy")
    for i in range(n_folds):
        print("Training on Fold: ", i+1)
        history=model_train(model_save_path, data_latih, label_data_latih, data_uji, label_data_uji, batch_size,epochs, i+1, alpha, model, objek)
        model_history.append(history)
        alpha = alpha * 0.1
    return model_history

jumlah_n_folds=2
folder_simpan_weight='processed_cnn/weight_hasil/'
Batch_Size = 4
Epochs=120
k_fold_training(jumlah_n_folds,folder_simpan_weight,Batch_Size,Epochs)