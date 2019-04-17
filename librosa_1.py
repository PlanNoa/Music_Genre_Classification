# import librosa
# import seaborn # optional
# import matplotlib.pyplot as plt
# import librosa.display
#
# x, sr = librosa.load('Music/a.wav')
#
# print(x.shape[0])
#
# logam = librosa.amplitude_to_db
# melgram = librosa.feature.melspectrogram
#
# print(logam)
# print(melgram)

from __future__ import print_function
from Code import freesound
import os

topic = input()

api_key = 'suGpSf4rteOn8Qnf1fNiWAb2rbp8WvayEvUhG7P5'
folder = './'+topic+'/'

freesound_client = freesound.FreesoundClient()
freesound_client.set_token(api_key)

try:
    os.mkdir(folder)
except:
    pass

print("Searching for '" + topic + "':\n")

results_pager = freesound_client.text_search(
    query=topic,
    sort="rating_desc",
    fields="id,name,previews,username"
)
print("Num results:", results_pager.count)


for page_idx in range(results_pager.count):
    if page_idx + 1 > 20: break
    print("\t----- PAGE", str(page_idx + 1), "-----")
    for sound in results_pager:
        print("\t-", sound.name, "by", sound.username)
        try:
            filename = str(sound.id) + '_' + sound.name.replace(u'/', '_') + ".wav"
            if not os.path.exists(folder + filename):
                sound.retrieve_preview(folder, filename)
        except:
            pass
    results_pager = results_pager.next_page()

    print()