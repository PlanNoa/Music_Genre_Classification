from sklearn.utils import shuffle
import numpy as np
import pickle
import os
from processor import compute_melgram

def getdata(tag):

    if os.path.exists("../Data/musics.pkl"):
        with open("../Data/musics.pkl", 'rb') as f:
            data = pickle.load(f)
        return data

    c = 1
    for genre in tag:
        if os.path.exists("../Data/" + genre + ".pkl"):
            pass
        else:
            X = []
            y = []
            file_list = os.listdir('../' + genre)
            for file in file_list:
                print("now processing:", genre + "-" + file)
                try:
                    X.append(compute_melgram('../' + genre + "/" + file))
                    y.append(c)
                except:
                    print("processing failed")
            with open("../Data/" + genre + ".pkl", 'wb') as f:
                data = [X, y]
                pickle.dump(data, f)
            c += 1

    all_X = []
    all_y = []
    data_list = os.listdir("../Data/")
    for file in data_list:
        with open("../Data/" + file, "rb") as f:
            temp = pickle.load(f)
        all_X.append(temp[0])
        all_y.append(temp[1])

    X = sum(all_X, [])
    y = sum(all_y, [])

    X, y = shuffle(X, y)
    data = [X, y]
    with open("../Data/musics.pkl", 'wb') as f:
        pickle.dump(data, f)

    return data
