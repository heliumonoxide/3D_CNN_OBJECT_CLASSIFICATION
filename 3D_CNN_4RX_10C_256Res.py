import os, sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential
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

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus, True)

# Documentation of 3D Convolution: https://keras.io/api/layers/convolution_layers/convolution3d/ and https://www.analyticsvidhya.com/blog/2022/05/building-a-3d-cnn-in-tensorflow/ 

def model_bryan():
    ##merancang model CNN##
    model1 = Sequential()

    model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="valid", input_shape=(256, 256, 256, 3)))
    # layer 1
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya
    model1.add(Conv3D(8, (3, 3, 3), padding='same', activation='relu'))  # dengan padding='same' output dari konvolusi hasilnya akan sama dengan ukuran inputnya, dengan cara menambahkan angka 0 di sisanya

    model1.add(
        MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))  # Mengambil nilai maksimum. bisa juga pool_sizenya 3x3 5x5 9x9
    
    # layer 2
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(Conv3D(16, (3, 3, 3), padding='same', activation='relu'))
    model1.add(MaxPooling3D(pool_size=(3, 3, 3), strides=(3, 3, 3)))

    # layer Hidden
    model1.add(Flatten())  # karena masih 2 dimensi, ubah ke 1 dimensi
    model1.add(Dense(20, input_dim=1, activation='relu'))
    model1.add(Dense(18, activation='relu'))
    model1.add(Dense(16, activation='relu'))
    model1.add(Dense(14, activation='relu'))
    model1.add(Dense(12, activation='relu'))
    model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # model1.add(Dense(10, activation='relu'))
    # layer Klasifikasi
    model1.add(Dense(4, activation='softmax'))  # karena ada 2 kelas yang diklasifikasikan. untuk activation, gunakan sigmoid jika hanya 2 kelas, jika lebih softmax

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
    model.compile(loss='binary_crossentropy', optimizer=opt,
                   metrics=['acc'])  # untuk 2 kelas, gunakan binary_crossentropy, bila lebih categorical_crossentropy


    # training model cnn
    start = time.time()
    history = model.fit(Train_data, Train_label, callbacks=[checkpoint], validation_data=(Valid_data, valid_label), epochs=Epochs,
                   batch_size=Batch_size)  # X= data, Y = label, epoch=jumlah pelatihan. batch_size= jumlah citra yang digunakan dalam 1 kali pembacaan
    end = time.time()

    # mencatat computational cost dalam bentuk .txt
    orig_stdout = sys.stdout
    nama_txt='processed_cnn/computational_cost/'+object+'_ComputationalCost_CNN2Rx_'+str(n_fold)+'.txt'
    f = open(nama_txt, 'w')
    sys.stdout = f
    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    sys.stdout = orig_stdout
    f.close()

    print("The time of execution of above program is :", (end - start) * 10 ** 3, "ms")
    plot_acc_lost(MODEL_SAVE_FOLDER_PATH+object+'/', history)
    print("\n training is done!! \n")

    # Print Validation Loss and Accuracy
    test_eval = model.evaluate(Valid_data, Valid_label)
    print('Test Loss: ', test_eval[0])
    print('Test Accuracy: ', test_eval[1])
    
    # predictions = model1.predict(testX, batch_size=32)
    #
    # #menampilkan confusion matrix data uji
    # confusion_matrix = metrics.confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1))
    # print(confusion_matrix)
    #
    # #membuat grafik confusion matrix data uji
    # x_axis_labels = ['Human','Cat','Peacock']
    # y_axis_labels = ['Human','Cat','Peacock']
    # sn.heatmap(confusion_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, fmt='d')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.tight_layout()
    # plt.savefig('ConfMatHumanCatPeacock_CNN2Rx.jpg')
    # plt.show()
    #
    # #membuat grafik confusion matrix data uji dalam persen
    # x_axis_labels = ['Human','Cat','Peacock']
    # y_axis_labels = ['Human','Cat','Peacock']
    # persen = confusion_matrix/np.sum(confusion_matrix,axis=1,keepdims=True).astype(float)
    # sn.heatmap(persen, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, fmt='.2%')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.tight_layout()
    # plt.savefig('ConfMatPersenHumanCatPeacock_CNN2Rx.jpg')
    # plt.show()
    #
    # #classification report
    # print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=['Human','Cat','Peacock'] ))
    #
    # #mencatat classification report dalam bentuk .txt
    # orig_stdout=sys.stdout
    # f = open('classreportHumanCatPeacock_CNN2Rx.txt','w')
    # sys.stdout = f
    # print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=['Human','Cat','Peacock'] ))
    # sys.stdout = orig_stdout
    # f.close()

    return history

def k_fold_training(n_folds,model_save_path,batch_size,epochs):
    model_history=[]
    alpha = 0.0001
    objek = "OU-MN-MT-Anoa"
    model = model_bryan()
    for i in range(n_folds):
        print("Training on Fold: ", i+1)
        data_latih = np.load("temp_train_test_data/trainX_256_4Class_OU-MN-MT-Anoa_10data.npy")
        label_data_latih = np.load("temp_train_test_data/trainY_256_4Class_OU-MN-MT-Anoa_10data.npy")
        data_uji = np.load("temp_train_test_data/testX_256_4Class_OU-MN-MT-Anoa_10data.npy")
        label_data_uji = np.load("temp_train_test_data/testY_256_4Class_OU-MN-MT-Anoa_10data.npy")
        history=model_train(model_save_path, data_latih, label_data_latih, data_uji, label_data_uji, batch_size,epochs, i+1, alpha, model, objek)
        model_history.append(history)
        # alpha = alpha * 0.1
    return model_history

jumlah_n_folds=1
folder_simpan_weight='processed_cnn/weight_hasil/'
Batch_Size = 1
Epochs=70
k_fold_training(jumlah_n_folds,folder_simpan_weight,Batch_Size,Epochs)