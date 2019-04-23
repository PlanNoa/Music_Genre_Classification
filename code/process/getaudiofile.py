from __future__ import print_function
from Code import freesound
import os

topic = input()

api_key = 'API-KEY'
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
