import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

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
    arraygabungan = []
    arraygabungan2 = []

    rx1 =  np.load(kelas_1_rx1)
    print("rx 1 done")

    rx2 = np.load(kelas_1_rx2)  # baca file data receiver pertama
    print("rx 2 done")

    rx3 = np.load(kelas_2_rx1)  # baca file data receiver pertama
    print("rx 3 done")

    rx4 = np.load(kelas_2_rx2)  # baca file data receiver pertama
    print("rx 4 done")

    count = 0
    ukuran_image_yang_diset = 256

    for k in range(1024):
        fft_rx2 = []  # buat array kosong untuk menampung data receiver kedua
        fft_rx1 = []  # buat array kosong untuk menampung data receiver pertama
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
    for i in range(1024):
        Label.append('Human')
        Label2.append('Bunny')

    GambarGabungan = np.concatenate((arraygabungan, arraygabungan2))
    LabelGabungan = np.concatenate((Label, Label2))

    # mengubah format array label ke dalam biner
    lb = MyLabelBinarizer()
    label = lb.fit_transform(LabelGabungan)
    # membagi data citra ke dalam data latih dan data uji
    # X itu gambar, Y itu label
    (trainX, testX, trainY, testY) = train_test_split(np.array(GambarGabungan), np.array(label), test_size=0.25)
    print(trainX.shape)
    print(testX.shape)

    return trainX, testX, trainY, testY

data_rx1='unpredicted_data/datafft_1_human_1mtr.npy'
data_rx2='unpredicted_data/datafft_2_human_1mtr.npy'
data_rx3='unpredicted_data/datafft_1_bunny_1mtr.npy'
data_rx4='unpredicted_data/datafft_2_bunny_1mtr.npy'
latihX, cobaX, latihY, cobaY = create_data_train_test(data_rx1, data_rx2, data_rx3, data_rx4)


np.save("temp_train_test_data/trainX_256_dummy.npy", latihX)
np.save("temp_train_test_data/testX_256_dummy.npy", cobaX)
np.save("temp_train_test_data/trainY_256_dummy.npy", latihY)
np.save("temp_train_test_data/testY_256_dummy.npy", cobaY)