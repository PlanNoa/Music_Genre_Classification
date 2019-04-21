# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.layers.convolutional import Conv2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
# from keras.layers import Input, Dense
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Reshape, Permute, Activation, merge, Embedding
# from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import ELU
# from keras.layers.recurrent import LSTM
# from keras.utils.data_utils import get_file
# import pickle
#
# np.random.seed(10)
#
# # def MusicTaggerCRNN(weights='msd', input_tensor=None):
# #
# #     # if K.image_dim_ordering() == 'th':
# #     #     input_shape = (1, 96, 1366)
# #     # else:
# #     #     input_shape = (96, 1366, 1)
# #     melgram_input = Input(shape=(1, 96, 1366))
# #     # channel_axis = 3
# #     # freq_axis = 1
# #     # time_axis = 2
# #
# #     # # x = ZeroPadding2D(padding=(0, 37))(melgram_input)
# #     # x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)
# #     #
# #     # x = Convolution2D(64, 3, 3, border_mode='same', name='conv1', trainable=False)(x)
# #     # x = BatchNormalization(axis=channel_axis, mode=0, name='bn1', trainable=False)(x)
# #     # x = ELU()(x)
# #     # # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1', trainable=False)(x)
# #     # x = Dropout(0.1, name='dropout1', trainable=False)(x)
# #     #
# #     # x = Convolution2D(128, 3, 3, border_mode='same', name='conv2', trainable=False)(x)
# #     # x = BatchNormalization(axis=channel_axis, mode=0, name='bn2', trainable=False)(x)
# #     # x = ELU()(x)
# #     # # x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2', trainable=False)(x)
# #     # x = Dropout(0.1, name='dropout2', trainable=False)(x)
# #     #
# #     # x = Convolution2D(128, 3, 3, border_mode='same', name='conv3', trainable=False)(x)
# #     # x = BatchNormalization(axis=channel_axis, mode=0, name='bn3', trainable=False)(x)
# #     # x = ELU()(x)
# #     # # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3', trainable=False)(x)
# #     # x = Dropout(0.1, name='dropout3', trainable=False)(x)
# #     #
# #     # x = Convolution2D(128, 3, 3, border_mode='same', name='conv4', trainable=False)(x)
# #     # x = BatchNormalization(axis=channel_axis, mode=0, name='bn4', trainable=False)(x)
# #     # x = ELU()(x)
# #     # # x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4', trainable=False)(x)
# #     # x = Dropout(0.1, name='dropout4', trainable=False)(x)
# #     #
# #     # # reshaping
# #     # if K.image_dim_ordering() == 'th':
# #     #     print(1)
# #     #     x = Permute((3, 1, 2))(x)
# #     # print(x)
# #     # # x = Reshape((1, 170, 128))(x)
# #     #
# #     # # GRU block 1, 2, output
# #     # # x = LSTM(32, return_sequences=True, name='gru1', input_shape=(96, 1366))(x)
# #     # # x = LSTM(32, return_sequences=False, name='gru2')(x)
# #     # # x = Dropout(0.3, name='final_drop')(x)
# #     #
# #     # if weights is None:
# #     #     # Create model
# #     #     print(x.shape)
# #     #     x = Dense(units = 10, activation='relu', name='output1')(x)
# #     #     x = Dense(units = 10, activation='softmax', name='output2')(x)
# #     #     print(x.shape)
# #     #     model = Model(melgram_input, x)
# #     #     return model
#
# model = Sequential()
# model.add(Dense(512, input_dim = 96 * 1366, activation='relu'))
# model.add(Dense(1, activation='softmax'))
#
# nb_classes = 10
# nb_epoch = 40
# batch_size = 100
#
# model_name = "crnn_net_adam_ours"
# model_path = "models_trained/" + model_name + "/"
# weights_path = "models_trained/" + model_name + "/weights/"
#
# time_elapsed = 0
#
# # model = MusicTaggerCRNN(weights=None, input_tensor=(1, 96, 1366))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# with open("../Data/musics_temp .pkl", 'rb') as f:
#     data = pickle.load(f)
#
# X = data[0]
# # for x in data[0]:
# #     X.append(x[0])
# Y = data[1]
# # for y in data[1]:
# #     Y.append([y])
#
# # print(np.array(X).shape)
# # print(np.array(Y).shape)
# # print(X[0])
# X, Y = np.array(X), np.array(Y)
# X_train = X[:int(len(X) * 0.8)]
# X_test = X[int(len(X) * 0.8):]
# Y_train = Y[:int(len(X) * 0.8)]
# Y_test = Y[int(len(X) * 0.8):]
# print(model.summary())
# print()
#
# print ("Training the model")
# for epoch in range(1,nb_epoch+1):
#     print ("Number of epoch: " +str(epoch)+"/"+str(nb_epoch))
#     scores = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1, verbose=1, validation_data=(X_test, Y_test))
#     print ("Time Elapsed: " +str(time_elapsed))
#
#     train = model.predict(X_train)
#     print(train)
#     # score_train = model.evaluate(X_train, Y_train, verbose=0)
#     # print('Train Loss:', score_train[0])
#     # print('Train Accuracy:', score_train[1])
#     #
#     # score_test = model.evaluate(X_test, Y_test, verbose=0)
#     # print('Test Loss:', score_test[0])
#     # print('Test Accuracy:', score_test[1])
#
#
#     if epoch % 5 == 0:
#         model.save_weights(weights_path + model_name + "_epoch_" + str(epoch) + ".h5")
#         print("Saved model to disk in: " + weights_path + model_name + "_epoch" + str(epoch) + ".h5")
#
#

import tensorflow as tf
import random
import pickle
import os
from sklearn.utils import shuffle
import numpy as np

X = tf.placeholder(tf.float32, [None, 1366])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape=[1366, 683],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([683]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[683, 1366],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([1366]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable("W3", shape=[1366, 683],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([683]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

W4 = tf.get_variable("W4", shape=[683, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

SAVER_DIR = "model"
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state("C:/Users/이희웅/PycharmProjects/librosa/Code/model/model-")

with open("../Data/musics_temp.pkl", 'rb') as f:
    data = pickle.load(f)

# X, Y = np.array(data[0]), np.array(data[1])
x, y = shuffle(data[0], data[1])
train_X = x[:int(len(x) * 0.8)]
test_X = x[int(len(x) * 0.8):]
train_Y = y[:int(len(y) * 0.8)]
test_Y = y[int(len(y) * 0.8):]

batch_size = 1000

for epoch in range(1000):
    avg_cost = 0
    total_batch = int(len(train_X) / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = train_X[total_batch * epoch :total_batch * (epoch + 1)], train_Y[total_batch * epoch :total_batch * (epoch + 1)]
        feed_dict = {X: batch_xs, Y: batch_ys}
        print("training -", str(i) + "/" + str(total_batch))
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print("training -", str(total_batch) + "/" + str(total_batch))

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    if epoch % 50 == 0:
        _accuracy = sess.run(accuracy, feed_dict={X: test_X, Y: test_Y})
        try:
            os.mkdir("C:/Users/이희웅/PycharmProjects/librosa/Code/model/model-" + str(_accuracy))
        except:
            pass
        saver.save(sess, "C:/Users/이희웅/PycharmProjects/librosa/Code/model/model-" + str(_accuracy) + "/model")
        print("accuracy (Restored) : %f" % _accuracy)

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: test_X, Y: test_Y}))
