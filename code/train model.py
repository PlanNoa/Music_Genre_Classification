# import os
# import time
# import sys
# from data import getdata
# import numpy as np
# from model import MusicTaggerCNN
#
# tag = np.array(['Alternative', 'Blues', 'Country', 'Hiphop', 'Jazz', 'Metal', 'POP', 'R&B', 'Reggae', 'Rock'])
#
# labelsDict = {
#     'Alternative':  1,
#     'Blues'     :   2,
#     'Country'   :   3,
#     'Hiphop'     :  4,
#     'Jazz'    :     5,
#     'Metal'      :  6,
#     'POP'     :     7,
#     'R&B'       :   8,
#     'Reggae'    :   9,
#     'Rock'      :   10,
# }
# data = getdata(tag)
# X = data[0]
# y = data[1]
#
# X_train = X[:int(len(X) * 0.8)]
# X_test = X[int(len(X) * 0.8):]
# y_train = np.array(y[:int(len(y) * 0.8)])
# y_test = np.array(y[int(len(y) * 0.8):])

from keras import backend as K
import os
import time
import h5py
import sys
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils
from math import floor
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import MusicTaggerCRNN
import pickle

# Parameters to set
TRAIN = 0
TEST = 1

SAVE_MODEL = 0
SAVE_WEIGHTS = 0

LOAD_MODEL = 0
LOAD_WEIGHTS = 1

# Dataset
MULTIFRAMES = 1
SAVE_DB = 0
LOAD_DB = 0

# Model parameters
nb_classes = 10
nb_epoch = 40
batch_size = 100

time_elapsed = 0

tags = np.array(['Alternative', 'Blues', 'Country', 'Hiphop', 'Jazz', 'Metal', 'POP', 'R&B', 'Reggae', 'Rock'])
tags = np.array(tags)

model_name = "crnn_net_adam_ours"
model_path = "models_trained/" + model_name + "/"
weights_path = "models_trained/" + model_name + "/weights/"

# Create directories for the models & weights
if not os.path.exists(model_path):
    os.makedirs(model_path)
    print ('Path created: ', model_path)

if not os.path.exists(weights_path):
    os.makedirs(weights_path)
    print ('Path created: ', weights_path)

# Initialize model
model = MusicTaggerCRNN(weights='msd', input_tensor=(1, 96, 1366))
#model = MusicTaggerCNN(weights='msd', input_tensor=(1, 96, 1366))
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

with open("../Data/musics.pkl", 'rb') as f:
    data = pickle.load(f)

X = data[0]
y = data[1]
print(len(X))
print(len(y))

X_train = X[:int(len(X) * 0.8)]
X_test = X[int(len(X) * 0.8):]
Y_train = np.array(y[:int(len(y) * 0.8)])
Y_test = np.array(y[int(len(y) * 0.8):])

# Train model
if TRAIN:
    try:
        print ("Training the model")
        for epoch in range(1,nb_epoch+1):
            print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
            scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))

            score_train = model.evaluate(X_train, Y_train, verbose=0)
            print('Train Loss:', score_train[0])
            print('Train Accuracy:', score_train[1])

            score_test = model.evaluate(X_test, Y_test, verbose=0)
            print('Test Loss:', score_test[0])
            print('Test Accuracy:', score_test[1])

            if SAVE_WEIGHTS and epoch % 5 == 0:
                model.save_weights(weights_path + model_name + "_epoch_" + str(epoch) + ".h5")
                print("Saved model to disk in: " + weights_path + model_name + "_epoch" + str(epoch) + ".h5")
    except:
        pass

# if TEST:
#     t0 = time.time()
#     print ('Predicting...','\n')
#
#     real_labels_mean = load_gt(test_gt_list)
#     real_labels_frames = y_test
#
#     results = np.zeros((X_test.shape[0], tags.shape[0]))
#     predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))
#     predicted_labels_frames = np.zeros((y_test.shape[0], 1))
#
#
#     song_paths = open(test_songs_list, 'r').read().splitlines()
#
#     previous_numFrames = 0
#     n=0
#     for i in range(0, num_frames_test.shape[0]):
#         print (song_paths[i])
#
#         num_frames=num_frames_test[i]
#         print ('Num_frames: ', str(num_frames),'\n')
#
#         results[previous_numFrames:previous_numFrames+num_frames] = model.predict(
#             X_test[previous_numFrames:previous_numFrames+num_frames, :, :, :])
#
#
#         for j in range(previous_numFrames,previous_numFrames+num_frames):
#             #normalize the results
#             total = results[j,:].sum()
#             results[j,:]=results[j,:]/total
#             sort_result(tags, results[j,:].tolist())
#
#             predicted_label_frames=predict_label(results[j,:])
#             predicted_labels_frames[n]=predicted_label_frames
#             n+=1
#
#
#         print ('\n',"Mean of the song: ")
#         results_song = results[previous_numFrames:previous_numFrames+num_frames]
#
#         mean=results_song.mean(0)
#         sort_result(tags, mean.tolist())
#
#         predicted_label_mean=predict_label(mean)
#
#         predicted_labels_mean[i]=predicted_label_mean
#         print ('\n','Predicted label: ', str(tags[predicted_label_mean]),'\n')
#
#         if predicted_label_mean != real_labels_mean[i]:
#             print ('WRONG!!')
#
#
#         previous_numFrames = previous_numFrames+num_frames
#
#         #break
#         print('\n\n\n')
#
#     cnf_matrix_frames = confusion_matrix(real_labels_frames, predicted_labels_frames)
#     plot_confusion_matrix(cnf_matrix_frames, classes=tags, title='Confusion matrix (frames)')
#
#     cnf_matrix_mean = confusion_matrix(real_labels_mean, predicted_labels_mean)
#     plot_confusion_matrix(cnf_matrix_mean, classes=tags, title='Confusion matrix (using mean)')