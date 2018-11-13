
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd


# In[2]:


LABELS = 10 # Number of different types of labels (1-10)
WIDTH = 28
EX_WIDTH = 24 # width / height of the image
CHANNELS = 1 # Number of colors in the image (greyscale)

VALID = 50000 # Validation data size

STEPS = 3500 #20000   # Number of steps to run
BATCH = 100 # Stochastic Gradient Descent batch size
PATCH = 5 # Convolutional Kernel size
DEPTH = 8 #32 # Convolutional Kernel depth size == Number of Convolutional Kernels
HIDDEN = 100 #1024 # Number of hidden neurons in the fully connected layer


# In[3]:


sess = tf.InteractiveSession()
data = pd.read_csv('D:\kaggle\Digit Recognizer\input\\train.csv')


# In[4]:


labels = np.array(data.pop('label'))
temp_labels = np.zeros((42000,10))
for i in range(42000):
    temp_labels[i][labels[i]] = 1.0
labels = temp_labels
data = data.values
data = data.reshape(-1,WIDTH,WIDTH,CHANNELS)


# In[5]:


permutation = np.random.permutation(labels.shape[0])
data = data[permutation,: ,: ,: ]
labels = labels[permutation,: ]


# In[6]:


ex_train_data = np.zeros((40000 * 25,24,24))
ex_valid_data = np.zeros((2000 * 25,24,24))
ex_train_data = ex_train_data.reshape(-1,EX_WIDTH,EX_WIDTH,CHANNELS)
ex_valid_data = ex_valid_data.reshape(-1,EX_WIDTH,EX_WIDTH,CHANNELS)
ex_train_labels = np.zeros((40000 * 25,10))
ex_valid_labels = np.zeros((2000 * 25,10))


# In[7]:


for i in range(40000):
    for j in range(5):
        for k in range(5):
            ex_train_data[25 * i + 5 * j + k,:,:,:] = data[i, j : (j + 24), k : (k + 24),:]
            ex_train_labels[25 * i + 5 * j + k,:] = labels[i,:]
for i in range(2000):
    for j in range(5):
        for k in range(5):
            ex_valid_data[25 * i + 5 * j + k,:,:,:] = data[40000 + i, j : (j + 24), k : (k + 24),:]
            ex_valid_labels[25 * i + 5 * j + k,:] = labels[40000 + i,:]


# In[8]:


PATCH1 = 3
HIDDEN = 64
DEPTH = 16

tf_data = tf.placeholder(tf.float32, shape=(None, EX_WIDTH, EX_WIDTH, CHANNELS))
tf_labels = tf.placeholder(tf.float32, shape=(None, LABELS))
conv_keep_prob = tf.placeholder('float')
keep_prob = tf.placeholder('float')

w1 = tf.Variable(tf.truncated_normal([PATCH, PATCH, CHANNELS, DEPTH], stddev=0.1))
b1 = tf.Variable(tf.zeros([DEPTH]))

w11 = tf.Variable(tf.truncated_normal([PATCH1, PATCH1, CHANNELS, DEPTH], stddev=0.1))
b11 = tf.Variable(tf.zeros([DEPTH]))

w2 = tf.Variable(tf.truncated_normal([PATCH, PATCH, DEPTH, 4 * DEPTH], stddev=0.1))
b2 = tf.Variable(tf.constant(1.0, shape=[4 * DEPTH]))

w21 = tf.Variable(tf.truncated_normal([PATCH1, PATCH1, DEPTH, 4 * DEPTH], stddev=0.1))
b21 = tf.Variable(tf.constant(1.0, shape=[4 * DEPTH]))

w3 = tf.Variable(tf.truncated_normal([6 * 6 * 4 * DEPTH, 4 * HIDDEN], stddev=0.1))
b3 = tf.Variable(tf.constant(1.0, shape=[4 * HIDDEN]))
w33 = tf.Variable(tf.truncated_normal([6 * 6 * 4 * DEPTH, 4 * HIDDEN], stddev=0.1))

w4 = tf.Variable(tf.truncated_normal([4 * HIDDEN, HIDDEN], stddev=0.1))
b4 = tf.Variable(tf.constant(1.0, shape=[HIDDEN]))

w5 = tf.Variable(tf.truncated_normal([HIDDEN, LABELS], stddev=0.1))
b5 = tf.Variable(tf.constant(1.0, shape=[LABELS]))

def logits(data):
    data = tf.nn.dropout(data, conv_keep_prob)
    
    # Convolutional layer 1
    x = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x + b1)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Convolutional layer 2
    x = tf.nn.conv2d(x, w2, [1, 1, 1, 1], padding='SAME')
    x = tf.nn.relu(x + b2)
    x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Convolutional layer 11
    y = tf.nn.conv2d(data, w11, [1, 1, 1, 1], padding='SAME')
    y = tf.nn.relu(y + b11)
    y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Convolutional layer 22
    y = tf.nn.conv2d(y, w21, [1, 1, 1, 1], padding='SAME')
    y = tf.nn.relu(y + b21)
    y = tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    # Fully connected layer
    x = tf.reshape(x,(-1, 6 * 6 * 4 * DEPTH))
    y = tf.reshape(y,(-1, 6 * 6 * 4 * DEPTH))
    x = tf.nn.relu(tf.matmul(x, w3) + b3 + tf.matmul(y, w33))
    
    # Fully connected layer2
    x = tf.nn.relu(tf.matmul(x, w4) + b4)
    
    return tf.matmul(x, w5) + b5

tf_logits = logits(tf_data)
tf_pred = tf.nn.softmax(tf.nn.dropout(tf_logits,keep_prob))

tf.stop_gradient(tf_labels)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf_logits,labels=tf_labels))
train_step = tf.train.RMSPropOptimizer(0.0001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(tf_pred,1),tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[12]:


import gc
del data
del labels
gc.collect()


# In[9]:


sess.run(tf.global_variables_initializer())


# In[14]:


valid_accuracy = 90.0
i = 1
permutation = np.random.permutation(ex_train_labels.shape[0])
permutation = permutation[:200]
current_data = ex_train_data[permutation]
current_labels = ex_train_labels[permutation]
while valid_accuracy < 95:
    i += 1
    j = i % 2000
    train_step.run(feed_dict={tf_data:current_data,tf_labels:current_labels,keep_prob:0.5,conv_keep_prob:0.97})
    permutation = np.random.permutation(ex_train_labels.shape[0])
    permutation = permutation[:200]
    current_data = ex_train_data[permutation]
    current_labels = ex_train_labels[permutation]
    if j == 0:
        train_accuracy = 0.0
        valid_accuracy = 0.0
        for k in range(500):
            train_accuracy += accuracy.eval(feed_dict={tf_data:ex_train_data[k * 100:(k + 1) * 100],                                                       tf_labels:ex_train_labels[k * 100:(k + 1) * 100],keep_prob:1.0,conv_keep_prob:1.0}) / 5
            valid_accuracy += accuracy.eval(feed_dict={tf_data:ex_valid_data[k * 100:(k + 1) * 100],                                                       tf_labels:ex_valid_labels[k * 100:(k + 1) * 100],keep_prob:1.0,conv_keep_prob:1.0}) / 5
        print("Step %d, Train.Acc %g%%, Val.Acc %g%%"%(i,train_accuracy,valid_accuracy))
print("Accuracy %g"%valid_accuracy)


# In[15]:


test_data = pd.read_csv('D:\kaggle\Digit Recognizer\input\\test.csv')
test_data = test_data.values
test_data = test_data.reshape(-1,WIDTH,WIDTH,CHANNELS)


# In[16]:


out_data = np.zeros((28000,EX_WIDTH,EX_WIDTH,1))
test_pred = np.zeros((28000,10))
for i in range(25):
    j = i // 5
    k = i % 5
    out_data = test_data[:, j : (j + 24), k : (k + 24),:]
    for j in range(280):
        test_pred[j * 100:(j + 1) * 100] += tf_pred.eval(feed_dict={tf_data:out_data[j * 100:(j + 1) * 100],keep_prob:1.0,conv_keep_prob:1.0})
test_labels = np.argmax(test_pred, axis=1)


# In[17]:


submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
submission.to_csv('submission.csv', index=False)
submission.tail()


# In[18]:


sess.close()

