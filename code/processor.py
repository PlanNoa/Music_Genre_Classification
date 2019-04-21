import librosa
import numpy as np
from math import floor

def compute_melgram(audio_path):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    hop_length = 256
    dura = 29.12

    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_fit = int(dura*sr)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(dura*sr) - n_sample,))))
    elif n_sample > n_sample_fit:
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    amplitude = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = amplitude(melgram(y=src, sr=sr, hop_length=hop_length, n_fft=N_FFT, n_mels=N_MELS) ** 2,)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret
