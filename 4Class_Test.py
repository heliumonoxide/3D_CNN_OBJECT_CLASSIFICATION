from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
import os, sys
import datetime as dt
from random import shuffle

objek = 'Human_Bunny'

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


def create_data_train_test(kelas_1_rx1,kelas_1_rx2,kelas_2_rx1,kelas_2_rx2):
    startPreprocessData = dt.datetime.now()
    arraygabungan = []
    arraygabungan2 = []

    rx1 = np.load(kelas_1_rx1)  # merubah format rx menjadi array
    print("rx 1 done")

    rx2 = np.load(kelas_1_rx2)  # merubah format rx menjadi array
    print("rx 2 done")

    rx3 = np.load(kelas_2_rx1)  # merubah format rx menjadi array
    print("rx 3 done")

    rx4 = np.load(kelas_2_rx2)  # merubah format rx menjadi array
    print("rx 4 done")

    # rx3 = pd.read_csv(kelas_2_rx1)  # baca file data receiver pertama
    # rx3 = np.array(rx3)  # merubah format rx menjadi array
    # print("rx 3 done")

    # rx4 = pd.read_csv(kelas_2_rx2)  # baca file data receiver pertama
    # rx4 = np.array(rx4)  # merubah format rx menjadi array
    # print("rx 4 done")

    count = 0
    ukuran_image_yang_diset = 256

    waktuInputData = dt.datetime.now() - startPreprocessData

    print("Waktu input data: ")
    print(waktuInputData)

    for k in range(5): # yang diambil hanya 10 frame tensor untuk data validasi
        fft_rx1 = []  # buat array kosong untuk menampung data receiver pertama
        fft_rx2 = []  # buat array kosong untuk menampung data receiver kedua
        fft_rx3 = []  # buat array kosong untuk menampung data receiver ketiga
        fft_rx4 = []  # buat array kosong untuk menampung data receiver keempat
        for i in range(ukuran_image_yang_diset):
            rx11 = rx1[0][i + count]  # membaca data receiver1 yang berada pada kolom pertama sumber data
            fft_rx1.append(rx11)  # memasukkan hasil pembacaan data receiver1 pada array yang telah disediakan
            rx22 = rx2[0][i + count]  # membaca data receiver2 yang berada pada kolom kedua sumber data
            fft_rx2.append(rx22)  # memasukkan hasil pembacaan data receiver2 pada array yang telah disediakan
            rx33 = rx3[0][i + count]  # membaca data receiver1 yang berada pada kolom pertama sumber data
            fft_rx3.append(rx33)  # memasukkan hasil pembacaan data receiver1 pada array yang telah disediakan
            rx44 = rx4[0][i + count]  # membaca data receiver2 yang berada pada kolom kedua sumber data
            fft_rx4.append(rx44)  # memasukkan hasil pembacaan data receiver2 pada array yang telah disediakan
        # fft_rx1 = np.array(fft_rx1)  # merubah format fft_rx1 menjadi array
        # fft_rx2 = np.array(fft_rx2)  # merubah format fft_rx2 menjadi array
        count = i

        #####buat image 1048x1048, sebanyak 2 layer#####
        array = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, 2])  # membuat array kosong, 2 layer
        array2 = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, 2])  # membuat array kosong, 2 layer
        # masukkan nilai dari receiver pertama pada layer pertama, dan nilai receiver kedua pada layer kedua
        for i in range(len(fft_rx1)):
            for j in range(len(fft_rx2)):
                array[i, j] = [fft_rx1[i], fft_rx2[j]]
                array2[i, j] = [fft_rx3[i], fft_rx4[j]]
        # cek ukuran array yang sudah terisi
        # print(array.shape)
        # print(array)
        arraygabungan.append(
            array)  # memasukkan array gabungan receiver pertama dan receiver kedua ke dalam array kosong yang telah disediakan
        arraygabungan2.append(
            array2)  # memasukkan array gabungan receiver pertama dan receiver kedua ke dalam array kosong yang telah disediakan
        # arraygabungan = np.array(arraygabungan)
    print("Done proses penyatuan array")
    Label = []
    Label2 = []
    for i in range(5):
        Label.append('Human')
        Label2.append('Bunny')

    GambarGabungan = np.concatenate((arraygabungan, arraygabungan2))
    LabelGabungan = np.concatenate((Label, Label2))

    ind = [i for i in range(len(GambarGabungan))]
    shuffle(ind)

    Gambar_Gabungan = GambarGabungan[ind]
    Label_Gabungan = LabelGabungan[ind]

    # mengubah format array label ke dalam biner
    lb = MyLabelBinarizer()
    label = lb.fit_transform(Label_Gabungan)
    # membagi data citra ke dalam data latih dan data uji
    # X itu gambar, Y itu label
    (trainX, testX, trainY, testY) = train_test_split(np.array(Gambar_Gabungan), np.array(label), test_size=0.7)
    print(trainX.shape)
    print(testX.shape)

    endPreprocessData = dt.datetime.now()

    print("Waktu Preprocess: ")
    print(endPreprocessData-startPreprocessData-waktuInputData)

    return trainX, testX, trainY, testY

def validate_data():
    data_rx1="unpredicted_data/datafft_1_human_1mtr.npy"
    data_rx2="unpredicted_data/datafft_2_human_1mtr.npy"
    data_rx3='unpredicted_data/datafft_1_human_1mtr.npy'
    data_rx4='unpredicted_data/datafft_2_human_1mtr.npy'
    latihX, cobaX, latihY, cobaY = create_data_train_test(data_rx1, data_rx2, data_rx3, data_rx4)

    waktuPrediksiAwal = dt.datetime.now()
    model_loaded = load_model("./processed_cnn/weight_hasil/1/"+ objek +"/best_weight.hdf5")
    predictions = model_loaded.predict(cobaX, batch_size=32)
    prediction = np.mean(predictions, axis=0)
    print("Perhitungan prediksi: ")
    print(" ")
    print(prediction[0])
    print(" ")
    print(prediction[1])
    print(" ")

    for i in range(len(prediction)):
        if prediction[i] > 0.6: prediction[i] = 1
        elif prediction[i] < 0.4: prediction[i] = 0

    # print("Perhitungan prediksi: ")
    # print(" ")
    # print(prediction[0])
    # print(" ")
    # print(prediction[1])
    # print(" ")

    if prediction[0] == 1 and prediction[1] == 0: prediksi_akhir = "Manusia"
    elif prediction[0] == 0 and prediction[1] == 1: prediksi_akhir = "Non-Manusia"

    waktuPrediksiAkhir = dt.datetime.now()
    waktuPrediksi = waktuPrediksiAkhir - waktuPrediksiAwal

    print('Directory data yang diolah: \n' + data_rx1 + '\n' + data_rx2 + '\n' + data_rx3 + '\n' + data_rx4)
    print('Hasil Prediksi: ' + prediksi_akhir)
    print("Waktu Prediksi: ")
    print(waktuPrediksi)

    return prediksi_akhir

validate_data()
