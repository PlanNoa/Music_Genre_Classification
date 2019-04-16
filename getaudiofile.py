from __future__ import print_function
import freesound  # $ git clone https://github.com/MTG/freesound-python
import os

api_key = 'suGpSf4rteOn8Qnf1fNiWAb2rbp8WvayEvUhG7P5'
folder = './Jazz/'

freesound_client = freesound.FreesoundClient()
freesound_client.set_token(api_key)

try:
    os.mkdir(folder)
except:
    pass

query = input()

print("Searching for '" + query + "':\n")

results_pager = freesound_client.text_search(
    query=query,
    sort="rating_desc",
    fields="id,name,previews,username"
)
print("Num results:", results_pager.count)

for page_idx in range(results_pager.count):
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