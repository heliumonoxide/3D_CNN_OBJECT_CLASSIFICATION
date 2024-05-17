import numpy as np

# Testaja

# =========================
# Binary Processing
# =========================

# Tentukan path ke file biner
file_path = 'unprocessed_data\\RusaTimur_1MTR_4RX_128Chirp_8Frame_4.bin'

# Spesifikasi konfigurasi sensor
num_AdcSamples = 256  # Ganti sesuai dengan jumlah sampel pada setiap saluran
num_adcBits = 16 
num_RX = 4 # ganti jika digunakan 4 receiver
num_Lanes = 2
isReal = 0
# =========================
# Tentukan tipe data yang digunakan untuk membaca file biner
# Misalnya, float32 untuk data floating point
dtype = np.int16

# Buka file biner
with open(file_path, 'rb') as f:
    # Baca data dari file dengan format non-interleaved
    data = np.fromfile(f, dtype=dtype)
    if(num_adcBits != 16):
        l_max = 2**(num_adcBits-1) -1
        data[data > l_max] -= 2**num_adcBits

file_Size = len(data)

# Sekarang 'data' adalah array numpy yang berisi data dari file biner dengan format non-interleaved
# Setiap baris mewakili satu sampel, dan setiap kolom mewakili satu saluran
# print(data)
# print(file_Size)

if isReal == 1:
    # Menghitung numChirps sesuai dengan rumus yang diberikan
    numChirps = int(file_Size / (num_AdcSamples * num_RX))

    # Membuat array LVDS berukuran fileSize yang diisi dengan nilai 0
    LVDS = np.zeros(file_Size)

    # Mereshape adcData sesuai dengan numADCSamples, numRx, dan numChirps
    LVDS = np.reshape(data, (num_AdcSamples * num_RX, numChirps))

    # Transpose array LVDS
    LVDS = LVDS.T

elif isReal == 0:
    # Menghitung numChirps sesuai dengan rumus yang diberikan
    numChirps = int(file_Size / (2 * num_AdcSamples * num_RX))

    # Membuat array LVDS berukuran fileSize/2 yang diisi dengan nilai 0
    LVDS = np.zeros(int(file_Size / 2), dtype = 'complex_')

    # Gabungkan bagian real dan imajiner ke dalam data kompleks
    # Membaca file: 2I diikuti oleh 2Q
    counter = 0
    for i in range(0, file_Size-2, 4):
        LVDS[counter] = data[i] + 1j * data[i + 2]
        LVDS[counter + 1] = data[i + 1] + 1j * data[i + 3]
        counter += 2

    # Membuat kolom untuk setiap chirp
    LVDS = np.reshape(LVDS, (numChirps, num_AdcSamples * num_RX))

    # Setiap baris adalah data dari satu chirp
    # LVDS = LVDS.T

    # print(LVDS)
    # print(len(LVDS))

# Membuat array adcData berukuran numRX x (numChirps * numADCSamples) yang diisi dengan nilai 0
data = np.zeros((num_RX, numChirps * num_AdcSamples), dtype = 'complex_')

# Mengatur ulang data per RX
for row in range(num_RX):
    for i in range(numChirps):
        data[row, (i * num_AdcSamples):(i * num_AdcSamples + num_AdcSamples)] = LVDS[i, (row * num_AdcSamples):(row * num_AdcSamples + num_AdcSamples)]
  
print(data)
print(len(data[1]))

# =========================
# FFT Processing
# =========================
L = len(data[1])
Fs = 5000 # Frequency sampling
T = 1/Fs        
t = np.arange(0, L) * T 

if(num_RX == 1):
    # Assign data receiver ke array numpy
    data_receiver_one = data[0]

    # FFT processing
    YOne = np.fft.fft(data_receiver_one)

    # Menghitung magnitude spektrum FFT dan membaginya dengan L
    HOne = np.abs(YOne/L)

    # Memotong bagian pertama dari H sesuai dengan panjang data L
    data_fft_one = HOne[:L]

    # Melakukan penyesuaian pada data_fft_per_receiver
    data_fft_one[1:-1] *= 2

    np.save("processed_data/datafft_RusaTimur_1_1mtr.npy", data_fft_one) # Ubah nama sesuai dengan nama objek

elif(num_RX == 2):
    # Assign data receiver ke array numpy
    data_receiver_one = np.array(data[0])
    data_receiver_two = np.array(data[1])

    # FFT processing
    YOne = np.fft.fft(data_receiver_one)
    YTwo = np.fft.fft(data_receiver_two)

    # Menghitung magnitude spektrum FFT dan membaginya dengan L
    HOne = np.abs(YOne/L)
    HTwo = np.abs(YTwo/L)

    # Memotong bagian pertama dari H sesuai dengan panjang data L
    data_fft_one = HOne[:L]
    data_fft_two = HTwo[:L]

    # Melakukan penyesuaian pada data_fft_per_receiver
    data_fft_one[1:-1] *= 2
    data_fft_two[1:-1] *= 2

    # Menyalin baris pertama data_fft_per_receiver ke baris kedua
    data_fft_one = np.vstack((data_fft_one, data_fft_one))
    data_fft_two = np.vstack((data_fft_two, data_fft_two))

    print(data_fft_one)

    np.save("processed_data/datafft_RusaTimur_1_1mtr.npy", data_fft_one) # Ubah nama sesuai dengan nama objek
    np.save("processed_data/datafft_RusaTimur_2_1mtr.npy", data_fft_two) # Ubah nama sesuai dengan nama objek

elif(num_RX == 4):
    # Assign data receiver ke array numpy
    data_receiver_one = np.array(data[0])
    data_receiver_two = np.array(data[1])
    data_receiver_three = np.array(data[2])
    data_receiver_four = np.array(data[3])

    # FFT processing
    YOne = np.fft.fft(data_receiver_one)
    YTwo = np.fft.fft(data_receiver_two)
    YThree = np.fft.fft(data_receiver_three)
    YFour = np.fft.fft(data_receiver_four)

    # Menghitung magnitude spektrum FFT dan membaginya dengan L
    HOne = np.abs(YOne/L)
    HTwo = np.abs(YTwo/L)
    HThree = np.abs(YThree/L)
    HFour = np.abs(YFour/L)

    # Memotong bagian pertama dari H sesuai dengan panjang data L
    data_fft_one = HOne[:L]
    data_fft_two = HTwo[:L]
    data_fft_three = HThree[:L]
    data_fft_four = HFour[:L]

    # Melakukan penyesuaian pada data_fft_per_receiver
    data_fft_one[1:-1] *= 2
    data_fft_two[1:-1] *= 2
    data_fft_three[1:-1] *= 2
    data_fft_four[1:-1] *= 2

    # Menyalin baris pertama data_fft_per_receiver ke baris kedua
    # data_fft_one = np.vstack((data_fft_one, data_fft_one))
    # data_fft_two = np.vstack((data_fft_two, data_fft_two))
    # data_fft_three = np.vstack((data_fft_three, data_fft_three))
    # data_fft_four = np.vstack((data_fft_four, data_fft_four))

    np.save("processed_data/datafft_RusaTimur_1_1mtr.npy", data_fft_one) # Ubah nama sesuai dengan nama objek
    np.save("processed_data/datafft_RusaTimur_2_1mtr.npy", data_fft_two) # Ubah nama sesuai dengan nama objek
    np.save("processed_data/datafft_RusaTimur_3_1mtr.npy", data_fft_three) # Ubah nama sesuai dengan nama objek
    np.save("processed_data/datafft_RusaTimur_4_1mtr.npy", data_fft_four) # Ubah nama sesuai dengan nama objek

else:
    raise ValueError("Sorry, receiver used is not compatible")

f = Fs * np.arange(0, L) / L


# =========================
# Data Preprocessing done
# =========================
