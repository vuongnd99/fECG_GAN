import pyedflib
import numpy as np
from scipy import signal, io 
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
import padasip as pa
import pandas as pd
import os

class DataUtils:

    def __init__(self) -> None:
        self.dataset = ["a&decgdb/", "nifecg/", "synt_ecg/"]
        self.record = '/RECORDS' # each dataset has a RECORDs file which contains file names
    """
    Read data from edf files. 3 Datasets contains:
    - Abdominal and Direct ECG: 5 signal files, each file contains:
        - referenced fetal signals
        - 4 channels of abdominal signals

    - Non-Invasive Fetal ECG: each recording contains:
        - 2 thoracic signals
        - 3 or 4 abdominal signals

    - Synthesized dataset: to be updated
    """
    def readData(self, dataset, path='data/'):
        
        if dataset == 'nifecg':
            dataset_path = path + dataset
            fileNames = dataset_path + self.record 
            with open(fileNames, 'rb') as f:
                files = f.readlines()
            aECG = np.zeros((n-1, f.getNSamples()[0], len(files)))
            fECG = np.zeros((1, f.getNSamples()[0]), len(files))
            for i, file in enumerate(files):
                signal_path = dataset_path + file[:-1]
                f = pyedflib.EdfReader(signal_path)
                n = f.signals_in_file                
                fECG[0, :, i] = f.readSignal(0)
                fECG[0, :, i] = scale(self.butter_bandpass_filter(fECG[0, :, i], 1, 100, 1000), axis=1)
                for j in np.arange(1, n):
                    aECG[j - 1, :, i] = f.readSignal(j)

                aECG[:, :, i] = scale(self.butter_bandpass_filter(aECG[:, :, i], 1, 100, 1000), axis=1)
                aECG[:, :, i] = signal.resample(aECG[:, :, i], int(aECG.shape[1] / 5), axis=1)
                fECG[:, :, i] = signal.resample(fECG[:, :, i], int(fECG.shape[1] / 5), axis=1)
            return aECG, fECG
        elif dataset == 'a&decgdb':
            dataset_path = path + dataset
            file_paths = []
            for fileName in os.listdir(dataset_path):
                file_paths.append(os.path.join(dataset_path, fileName))
            n = len(file_paths)
            fECG = np.zeros((1, 10001, n))
            aECG = np.zeros((4, 10001, n))
            for i, file_path in enumerate(file_paths):
                data = pd.read_csv(file_path)
                fECG[:, :, i] = data[1]
                for j in range(4):
                    aECG[j, : , i] = data[j+2]
                fECG[0, :, i] = scale(self.butter_bandpass_filter(fECG[0, :, i], 1, 100, 1000), axis=1)
                aECG[:, :, i] = scale(self.butter_bandpass_filter(aECG[:, :, i], 1, 100, 1000), axis=1)
                aECG[:, :, i] = signal.resample(aECG[:, :, i], int(aECG.shape[1] / 5), axis=1)
                fECG[:, :, i] = signal.resample(fECG[:, :, i], int(fECG.shape[1] / 5), axis=1)
            return aECG, fECG
        else:
            dataset_path = path + 'fecgsyndb.mat'
            f = io.loadmat(dataset_path)
            
            
    def windowingSig(self, sig1, sig2, windowSize = 15):
        signalLen = sig2.shape[1]
        signalWindow1 = [sig1[:, int(i):int(i+windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
        signalWindow2 = [sig2[:, int(i):int(i+windowSize)].transpose() for i in range(0, signalLen - windowSize, windowSize)]
        return signalWindow1, signalWindow2

    def adaptFilter(self, src, ref):
        f = pa.filters.FilterNLMS(n=3, mu=0.1, w='random')
        for index, sig in enumerate(src):
            try:
                y, e, w = f.run(ref[index][:, 0], sig)
                ref[index][:, 0] = e
            except:
                pass
        return ref
    
    def calICA(self, sdSig, component=7):
        ica = FastICA(n_components=component, max_iter=1000)
        icaRes = []
        for index, sig in enumerate(sdSig):
            try:
                icaSignal = np.array(ica.fit_transform(sig))
                icaSignal = np.append(icaSignal, sig[:, range(2, 4)], axis=1)
                icaRes.append(icaSignal)
            except:
                pass
        return np.array(icaRes)

    def createDelayRepetition(self, signal, numberDelay=4, delay=10):
        signal = np.repeat(signal, numberDelay, axis=0)
        for row in range(1, signal.shape[0]):
            signal[row, :] = np.roll(signal[row, :], shift=delay*row)
        return signal

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3, axis=1):
        b, a = self.__butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data, axis=axis)
        return y

