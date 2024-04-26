import numpy as np
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


def create_data_train_test(kelas_1_rx1,kelas_1_rx2, kelas_1_rx3,kelas_2_rx1,kelas_2_rx2,kelas_2_rx3,kelas_3_rx1,kelas_3_rx2,kelas_3_rx3,kelas_4_rx1,kelas_4_rx2,kelas_4_rx3):
    arraygabungan = []
    arraygabungan2 = []
    arraygabungan3 = []
    arraygabungan4 = []

    class1_rx1 =  np.load(kelas_1_rx1)
    print("kelas 1 rx 1 done")

    class1_rx2 = np.load(kelas_1_rx2)
    print("kelas 1 rx 2 done")

    class1_rx3 = np.load(kelas_1_rx3)
    print("kelas 1 rx 3 done")

    class2_rx1 = np.load(kelas_2_rx1)
    print("kelas 2 rx 1 done")
    
    class2_rx2 = np.load(kelas_2_rx2)
    print("kelas 2 rx 2 done")
    
    class2_rx3 = np.load(kelas_2_rx3)
    print("kelas 2 rx 3 done")
    
    class3_rx1 = np.load(kelas_3_rx1)
    print("kelas 3 rx 1 done")
    
    class3_rx2 = np.load(kelas_3_rx2)
    print("kelas 3 rx 2 done")
    
    class3_rx3 = np.load(kelas_3_rx3)
    print("kelas 3 rx 3 done")
    
    class4_rx1 = np.load(kelas_4_rx1)
    print("kelas 4 rx 1 done")
    
    class4_rx2 = np.load(kelas_4_rx2)
    print("kelas 4 rx 2 done")
    
    class4_rx3 = np.load(kelas_4_rx3)
    print("kelas 4 rx 3 done")

    count = 0
    ukuran_image_yang_diset = 256

    for k in range(25):
        print("Tensor data ke: ")
        print(k+1)
        fft1_rx1 = []  # buat array kosong untuk menampung data receiver pertama kelas 1
        fft1_rx2 = []  # buat array kosong untuk menampung data receiver kedua kelas 1
        fft1_rx3 = []  # buat array kosong untuk menampung data receiver ketiga kelas 1
        fft2_rx1 = []  # buat array kosong untuk menampung data receiver pertama kelas 2
        fft2_rx2 = []  # buat array kosong untuk menampung data receiver kedua kelas 2
        fft2_rx3 = []  # buat array kosong untuk menampung data receiver ketiga kelas 2
        fft3_rx1 = []  # buat array kosong untuk menampung data receiver pertama kelas 3
        fft3_rx2 = []  # buat array kosong untuk menampung data receiver kedua kelas 3
        fft3_rx3 = []  # buat array kosong untuk menampung data receiver ketiga kelas 3
        fft4_rx1 = []  # buat array kosong untuk menampung data receiver pertama kelas 4
        fft4_rx2 = []  # buat array kosong untuk menampung data receiver kedua kelas 4
        fft4_rx3 = []  # buat array kosong untuk menampung data receiver ketiga kelas 4

        for i in range(ukuran_image_yang_diset):
            rx11 = class1_rx1[i + count]  
            fft1_rx1.append(rx11)  
            rx12 = class1_rx2[i + count]  
            fft1_rx2.append(rx12)
            rx13 = class1_rx3[i + count]  
            fft1_rx3.append(rx13)  
            rx21 = class2_rx1[i + count]  
            fft2_rx1.append(rx21)  
            rx22 = class2_rx2[i + count]  
            fft2_rx2.append(rx22)  
            rx23 = class2_rx3[i + count]  
            fft2_rx3.append(rx23)
            rx31 = class3_rx1[i + count]  
            fft3_rx1.append(rx31)  
            rx32 = class3_rx2[i + count]  
            fft3_rx2.append(rx32)  
            rx33 = class3_rx3[i + count]  
            fft3_rx3.append(rx33)  
            rx41 = class4_rx1[i + count]  
            fft4_rx1.append(rx41)  
            rx42 = class4_rx2[i + count]  
            fft4_rx2.append(rx42)  
            rx43 = class4_rx3[i + count]
            fft4_rx3.append(rx43)  
            
        # ======== Done for loop and saving last i into count =======
        count = i

        #####buat data tensor kubus256x256x256#####
        array_kelas1 = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, ukuran_image_yang_diset, 3])  # membuat tensor 3D kosong berbentuk kubus dengan dimensi 256
        array_kelas2 = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, ukuran_image_yang_diset, 3])  # membuat tensor 3D kosong berbentuk kubus dengan dimensi 256
        array_kelas3 = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, ukuran_image_yang_diset, 3])  # membuat tensor 3D kosong berbentuk kubus dengan dimensi 256
        array_kelas4 = np.zeros([ukuran_image_yang_diset, ukuran_image_yang_diset, ukuran_image_yang_diset, 3])  # membuat tensor 3D kosong berbentuk kubus dengan dimensi 256

        # masukkan nilai dari receiver pertama pada layer pertama, dan nilai receiver kedua pada layer kedua
        for i in range(ukuran_image_yang_diset):
            for j in range(ukuran_image_yang_diset):
                for k in range(ukuran_image_yang_diset):
                    array_kelas1[i, j, k] = [fft1_rx1[i], fft1_rx2[j], fft1_rx3[k]]
                    array_kelas2[i, j, k] = [fft2_rx1[i], fft2_rx2[j], fft2_rx3[k]]
                    array_kelas3[i, j, k] = [fft3_rx1[i], fft3_rx2[j], fft3_rx3[k]]
                    array_kelas4[i, j, k] = [fft4_rx1[i], fft4_rx2[j], fft4_rx3[k]]
        # cek ukuran array yang sudah terisi
        # print(array.shape)
        # print(array)
        arraygabungan.append(
            array_kelas1)
        arraygabungan2.append(
            array_kelas2)
        arraygabungan3.append(
            array_kelas3)
        arraygabungan4.append(
            array_kelas4)

        array_kelas1 = None
        array_kelas2 = None
        array_kelas3 = None
        array_kelas4 = None
        
    print("Done proses penyatuan array")
    del array_kelas1
    del array_kelas2
    del array_kelas3
    del array_kelas4

    Label = []
    Label2 = []
    Label3 = []
    Label4 = []

    for i in range(25):
        Label.append('Anoa')
        Label2.append('Makaka_Nemestrina')
        Label3.append('Makaka_Tonkeana')
        Label4.append('OrangUtan')

    GambarGabungan = np.concatenate((arraygabungan, arraygabungan2, arraygabungan3, arraygabungan4))
    LabelGabungan = np.concatenate((Label, Label2, Label3, Label4))

    del arraygabungan
    del arraygabungan2
    del arraygabungan3
    del arraygabungan4

    # mengubah format array label ke dalam biner
    lb = MyLabelBinarizer()
    label = lb.fit_transform(LabelGabungan)
    # membagi data citra ke dalam data latih dan data uji
    # X itu gambar, Y itu label
    (trainX, testX, trainY, testY) = train_test_split(np.array(GambarGabungan), np.array(label), test_size=0.2)
    print(trainX.shape)
    print(testX.shape)

    del GambarGabungan

    return trainX, testX, trainY, testY

class1_data_rx1='processed_data/datafft_Anoa_1_1mtr.npy'
class1_data_rx2='processed_data/datafft_Anoa_2_1mtr.npy'
class1_data_rx3='processed_data/datafft_Anoa_3_1mtr.npy'
class2_data_rx1='processed_data/datafft_Makaka_Nemestrina_1_1mtr.npy'
class2_data_rx2='processed_data/datafft_Makaka_Nemestrina_2_1mtr.npy'
class2_data_rx3='processed_data/datafft_Makaka_Nemestrina_3_1mtr.npy'
class3_data_rx1='processed_data/datafft_Makaka_Tonkeana_1_1mtr.npy'
class3_data_rx2='processed_data/datafft_Makaka_Tonkeana_2_1mtr.npy'
class3_data_rx3='processed_data/datafft_Makaka_Tonkeana_3_1mtr.npy'
class4_data_rx1='processed_data/datafft_OrangUtan_1_1mtr.npy'
class4_data_rx2='processed_data/datafft_OrangUtan_2_1mtr.npy'
class4_data_rx3='processed_data/datafft_OrangUtan_3_1mtr.npy'
latihX, cobaX, latihY, cobaY = create_data_train_test(class1_data_rx1, class1_data_rx2, class1_data_rx3, class2_data_rx1, class2_data_rx2, class2_data_rx3, class3_data_rx1, class3_data_rx2, class3_data_rx3, class4_data_rx1, class4_data_rx2, class4_data_rx3)


np.save("temp_train_test_data/trainX_256_4Class_OU-MN-MT-Anoa_25data.npy", latihX)
np.save("temp_train_test_data/trainY_256_4Class_OU-MN-MT-Anoa_25data.npy", latihY)
np.save("temp_train_test_data/testX_256_4Class_OU-MN-MT-Anoa_25data.npy", cobaX)
np.save("temp_train_test_data/testY_256_4Class_OU-MN-MT-Anoa_25data.npy", cobaY)