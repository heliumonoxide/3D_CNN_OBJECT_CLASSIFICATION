import sys
from sklearn import metrics
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import datetime as dt
import seaborn as sn
import matplotlib.pyplot as plt

objek = 'OU-MN-MT-Anoa'
target = ['Anoa','Makaka_Nemestrina','Makaka_Tonkeana','OrangUtan']


def validate_data():
    data_test = "./temp_train_test_data/trainX_256_4Class_OU-MN-MT-Anoa_test.npy"
    output_test = "./temp_train_test_data/trainY_256_4Class_OU-MN-MT-Anoa_test.npy"
    cobaX = np.load(data_test)
    cobaY = np.load(output_test)

    waktuPrediksiAwal = dt.datetime.now()
    model_loaded = load_model("./processed_cnn/weight_hasil/"+ objek +"/1/best_weight.hdf5")
    predictions = model_loaded.predict(cobaX, batch_size=1)

    #menampilkan confusion matrix data uji
    confusion_matrix = metrics.confusion_matrix(cobaY.argmax(axis=1),predictions.argmax(axis=1))
    
    #membuat grafik confusion matrix data uji
    x_axis_labels = target
    y_axis_labels = target
    sn.heatmap(confusion_matrix, xticklabels=x_axis_labels, yticklabels=y_axis_labels, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('./processed_cnn/figures/ConfMat'+objek+'_CNN3Rx.jpg')
    # plt.show()

    #classification report
    print(metrics.classification_report(cobaY.argmax(axis=1),predictions.argmax(axis=1), target_names=target ))
    
    #mencatat classification report dalam bentuk .txt
    orig_stdout=sys.stdout
    f = open('./processed_cnn/reports/classreport'+objek+'_CNN3Rx.txt','w')
    sys.stdout = f
    print(metrics.classification_report(cobaY.argmax(axis=1),predictions.argmax(axis=1), target_names=target ))
    sys.stdout = orig_stdout
    f.close()

    del predictions

    print("\n\nHasil Evaluasi: ")
    evaluations = model_loaded.evaluate(cobaX, cobaY, batch_size=1)
    print(evaluations)

    waktuPrediksiAkhir = dt.datetime.now()
    waktuPrediksi = waktuPrediksiAkhir - waktuPrediksiAwal

    print("Objek : " + objek)
    print("Waktu Prediksi: ")
    print(waktuPrediksi)

    return evaluations

validate_data()
