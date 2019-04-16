import librosa
import numpy as np
from math import floor

SR = 12000
N_FFT = 512
N_MELS = 96
hop_length = 256
dura = 29.12

def compute_melgram(audio_path):

    src, sr = librosa.load(audio_path, sr=SR)
    n_sample = src.shape[0]
    n_sample_fit = int(dura*sr)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(dura*sr) - n_sample,))))
    elif n_sample > n_sample_fit:
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    amplitude = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = amplitude(melgram(y=src, sr=sr, hop_length=hop_length, n_fft=N_FFT, n_mels=N_MELS) ** 2,)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret

print(compute_melgram("./Music/a.wav"))

# def compute_melgram_multiframe(audio_path, all_song=True):
#
#     SR = 12000
#     N_FFT = 512
#     N_MELS = 96
#     HOP_LEN = 256
#     DURA = 29.12  # to make it 1366 frame..
#     if all_song:
#         DURA_TRASH = 0
#     else:
#         DURA_TRASH = 20
#
#     src, sr = librosa.load(audio_path, sr=SR)  # whole signal
#     n_sample = src.shape[0]
#     n_sample_fit = int(DURA * SR)
#     n_sample_trash = int(DURA_TRASH * SR)
#
#     # remove the trash at the beginning and at the end
#     src = src[n_sample_trash:(n_sample - n_sample_trash)]
#     n_sample = n_sample - 2 * n_sample_trash
