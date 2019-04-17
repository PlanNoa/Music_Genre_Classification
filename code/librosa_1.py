import librosa
import seaborn # optional
import matplotlib.pyplot as plt
import librosa.display

x, sr = librosa.load('Music/a.wav')

print(x.shape[0])

logam = librosa.amplitude_to_db
melgram = librosa.feature.melspectrogram

print(logam)
print(melgram)
