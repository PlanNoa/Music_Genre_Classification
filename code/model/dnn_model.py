import tensorflow as tf
import random
import pickle
import os
from sklearn.utils import shuffle
from data import getdata
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

SAVER_DIR = "Your DIR"
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

tags = np.array(['Alternative', 'Blues', 'Country', 'Hiphop', 'Jazz', 'Metal', 'POP', 'RnB', 'Reggae', 'Rock'])
data = getdata(tags)

x, y = shuffle(data[0], data[1])
train_X = x[:int(len(x) * 0.8)]
test_X = x[int(len(x) * 0.8):]
train_Y = y[:int(len(y) * 0.8)]
test_Y = y[int(len(y) * 0.8):]

batch_size = 1000

for epoch in range(30):
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
            os.mkdir(SAVER_DIR + "/model-" + str(_accuracy))
        except:
            pass
        saver.save(sess, SAVER_DIR + "/model-" + str(_accuracy) + "/model")
        print("accuracy (Restored) : %f" % _accuracy)

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: test_X, Y: test_Y}))
