import tensorflow as tf
import numpy as np
import pickle

def getBatch(data, labels, batchSize, iteration):
    startOfBatch = (iteration * batchSize) % len(data)
    endOfBacth = (iteration * batchSize + batchSize) % len(data)

    if startOfBatch < endOfBacth:
        return data[startOfBatch:endOfBacth], labels[startOfBatch:endOfBacth]
    else:
        dataBatch = np.vstack((data[startOfBatch:],data[:endOfBacth]))
        labelsBatch = np.vstack((labels[startOfBatch:],labels[:endOfBacth]))

        return dataBatch, labelsBatch


if __name__ == "__main__":
    np.random.seed(777)
    learning_rate = 0.001
    training_iters = 100000
    batch_size = 64
    display_step = 1
    train_size = 800

    n_input = 96 * 1366
    n_classes = 10
    dropout = 0.9
    best_acc = 0.0

    with open("C:/Users/user/Data/musics_f.pkl", 'rb') as f:
        _data = pickle.load(f)

    data = _data[0]
    labels = _data[1]
    data = data.reshape((1000, 96, 1366, -1))
    print(data.shape)

    trainData = data[:train_size]
    trainLabels = labels[:train_size]

    testData = data[train_size:]
    testLabels = labels[train_size:]

    x = tf.placeholder(tf.float32, [None, 96, 1366, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)


    def conv2d(sound, w, b):
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1],
                                                      padding='SAME'), b))


    def max_pool(sound, k):
        return tf.nn.max_pool(sound, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


    def conv_net(_X, _weights, _biases, _dropout):
        _X = tf.reshape(_X, shape=[-1, 96, 1366, 1])
        conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
        conv1 = max_pool(conv1, k=4)
        conv1 = tf.nn.dropout(conv1, _dropout)

        conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
        conv2 = max_pool(conv2, k=2)
        conv2 = tf.nn.dropout(conv2, _dropout)

        conv3 = conv2d(conv2, _weights['wc3'], _biases['bc3'])
        conv3 = max_pool(conv3, k=2)
        conv3 = tf.nn.dropout(conv3, _dropout)

        dense1 = tf.reshape(conv3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1']))
        dense1 = tf.nn.dropout(dense1, _dropout)

        out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
        return out


    weights = {
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 102])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 102, 73])),
        'wc3': tf.Variable(tf.random_normal([4, 4, 73, 35])),
        'wd1': tf.Variable(tf.random_normal([18060, 8192])),
        'out': tf.Variable(tf.random_normal([8192, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([102])+0.01),
        'bc2': tf.Variable(tf.random_normal([73])+0.01),
        'bc3': tf.Variable(tf.random_normal([35])+0.01),
        'bd1': tf.Variable(tf.random_normal([8192])+0.01),
        'out': tf.Variable(tf.random_normal([n_classes])+0.01)
    }

    pred = conv_net(x, weights, biases, keep_prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.initialize_all_variables()

    SAVER_DIR = "C:/Users/user/model/"
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        step = 1
        while step * batch_size < training_iters:
            batch_xs, batch_ys = getBatch(trainData, trainLabels, batch_size, step)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            try:
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    print("Iter " + str(step * batch_size) + ", Loss= " + "{:.6f}".format(loss) + ", Accuracy= " + "{:.5f}".format(acc))
                    if acc > best_acc:
                        best_acc = acc
                        save_path = saver.save(sess, SAVER_DIR + "model-" + str(acc) + ".ckpt")
            except:
                print("save failed")
            step += 1
        print("Optimization Finished!")

        save_path = saver.save(sess, SAVER_DIR + "model.final")
        print("Model saved in file: %s" % save_path)

        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: testData,
                                                                 y: testLabels,
                                                                 keep_prob: 1.}))
