import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
import padasip as pa

class DataUtils:

    def __init__(self) -> None:
        self.fileNames = ["r01.edf", "r04.edf", "r07.edf", "r08.edf", "r10.edf"]

    def readData(self, sigNum, path):
        file_name = path + self.fileNames[sigNum]
        f = pyedflib.EdfReader(file_name)
