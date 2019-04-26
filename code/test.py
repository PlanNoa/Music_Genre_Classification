import librosa
import numpy as np
import tensorflow as tf

def compute_melgram(src, sr):
    N_FFT = 512
    N_MELS = 96
    hop_length = 256
    dura = 29.12

    n_sample = src.shape[0]
    n_sample_fit = int(dura*sr)

    if n_sample < n_sample_fit:
        src = np.hstack((src, np.zeros((int(dura*sr) - n_sample,))))
    elif n_sample > n_sample_fit:
        src = src[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]
    amplitude = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = amplitude(melgram(y=src, sr=sr, hop_length=hop_length, n_fft=N_FFT, n_mels=N_MELS) ** 2,)
    ret = ret[np.newaxis, np.newaxis, :]
    return ret

x = tf.placeholder(tf.float32, [None, 96, 1366, 1])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

def conv2d(sound, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(sound, w, strides=[1, 1, 1, 1], padding='SAME'), b))

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
    dense1 = tf.nn.dropout(dense1, _dropout)  # Apply Dropout

    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

weights = {
        'wc1': tf.Variable(tf.random_normal([4, 4, 1, 102])),
        'wc2': tf.Variable(tf.random_normal([4, 4, 102, 73])),
        'wc3': tf.Variable(tf.random_normal([4, 4, 73, 35])),
        'wd1': tf.Variable(tf.random_normal([18060, 8192])),
        'out': tf.Variable(tf.random_normal([8192, 10]))
    }

biases = {
        'bc1': tf.Variable(tf.random_normal([102])+0.01),
        'bc2': tf.Variable(tf.random_normal([73])+0.01),
        'bc3': tf.Variable(tf.random_normal([35])+0.01),
        'bd1': tf.Variable(tf.random_normal([8192])+0.01),
        'out': tf.Variable(tf.random_normal([10])+0.01)
    }

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


cut = 360161
wav = "C:/Users/이희웅/Music/Evolve/Imagine Dragons Dancing In The Dark Ft Dawn Foxes Music.mp3"
src, sr = librosa.load(wav, 12000)

data = []

for c in range(0, len(src), int(cut/2)):
    try:
        data.append(compute_melgram(src[c:c+cut], sr).reshape(-1, 96, 1366, 1))
    except:
        pass

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('C:/Users/이희웅/PycharmProjects/librosa/Code/model/model-real')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("fail")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

result = []
for d in data:
    _pred = sess.run(pred, feed_dict={x: d, keep_prob:1.})
    for p in _pred:
        result.append(softmax(p))

print(sum(result))